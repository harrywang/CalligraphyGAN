import numpy as np
import cv2
import math
import random
import threading
import os
from utils import cv_read_img_BGR as imread
import threadpool as tp


# "ReLU"
def lc(x):
    return int(max(0, x))


def generate_motion_blur_kernel(dim=3, angle=0., threshold_factor=1.3, divide_by_dim=True):
    radian = angle / 360 * math.pi * 2 + math.pi / 2

    # first, generate xslope and yslope
    gby, gbx = np.mgrid[0:dim, 0:dim]
    cen = (dim + 1) / 2 - 1
    gbx = gbx - float(cen)
    gby = gby - float(cen)

    # then mix the slopes according to angle
    gbmix = gbx * math.cos(radian) - gby * math.sin(radian)

    kernel = (threshold_factor - gbmix * gbmix).clip(min=0., max=1.)

    if divide_by_dim:
        kernel /= dim * dim * np.mean(kernel)

    return kernel.astype('float32')


def sigmoid_array(x):
    sgm = 1. / (1. + np.exp(-x))
    return np.clip(sgm * 1.2 - .1, a_max=1., a_min=0.)


def rn():
    return random.random()


def limit(x, minimum, maximum):
    return min(max(x, minimum), maximum)


# Brush Factory
class BrushFactory:
    def __init__(self, brush_dir='./brushes'):
        self.brush_dir = brush_dir
        self.brushes = {}
        self._load_brush()

    def _load_brush(self):
        for fn in os.listdir(self.brush_dir):
            if os.path.isfile(os.path.join(self.brush_dir, fn)):
                brush = cv2.imread(os.path.join(self.brush_dir, fn), 0)
                if brush is not None:
                    self.brushes[fn] = brush

    def get_brush(self, key='random'):
        if key == 'random':
            key = random.choice(list(self.brushes.keys()))
        brush = self.brushes[key]
        return brush, key

    def rotate_brush(self, brush, rad, srad, angle):
        # brush image should be of grayscale, pointing upwards
        # translate w x h into an area of 2rad x 2rad

        bh, bw = brush.shape[0:2]
        # print(brush.shape)

        osf = 0.1
        # oversizefactor: ratio of dist-to-edge to width,
        # to compensate for the patch smaller than the original ellipse

        rad = int(rad * (1. + osf))
        srad = int(srad * (1. + osf))

        # 1. scale
        orig_points = np.array([[bw / 2, 0], [0, bh / 2], [bw, bh / 2]]).astype('float32')
        # x,y of top left right
        translated = np.array([[rad, 0], [rad - srad, rad], [rad + srad, rad]]).astype('float32')

        # affine transform matrix
        at = cv2.getAffineTransform(orig_points, translated)

        at = np.vstack([at, [0, 0, 1.]])

        # 2. rotate
        rm = cv2.getRotationMatrix2D((rad, rad), angle - 90, 1)
        rm = np.vstack([rm, [0, 0, 1.]])

        # 3. combine 2 affine transform
        cb = np.dot(rm, at)

        # 4. do the transform
        res = cv2.warpAffine(brush, cb[0:2, :], (rad * 2, rad * 2))
        return res


class OilPaint:
    def __init__(self, image, target_color=None):
        """

        :param image: cv2 format Image data
        :param target_color: the value is between [0., 1.]
        """
        if target_color is not None:
            self.target_color = list(target_color)
        else:
            self.target_color = [[0., 0., 0.]]

        # convert to float32
        original_image = image.astype('float32') / 255

        # canvas initialized
        canvas = original_image.copy()
        # canvas[:, :] = 0.94117647  # initialize canvas with white
        canvas[:, :] = 1.

        self.canvas = canvas
        self.original_image = original_image

        self.canvas_lock = threading.Lock()
        self.canvas_lock.acquire()
        self.canvas_lock.release()

        self.bf = BrushFactory(brush_dir='./brushes')

        self.bp_filter = np.array([13, 3, 7.]).astype('float32')

        self.color_delta = 1e-4
        self.angle_delta = 5.
        self.x_delta = 2.
        self.y_delta = 2.
        self.radius_delta = 5.

    def _positive_sharpen(self, i, over_blur=False, coeff=8.):  # no darken to original image
        # emphasize the edges d
        blurred = cv2.blur(i, (5, 5))
        sharpened = i + (i - blurred) * coeff
        if over_blur:
            return cv2.blur(np.maximum(sharpened, i), (11, 11))
        return cv2.blur(np.maximum(sharpened, i), (3, 3))

    def _diff(self, i1, i2, over_blur=False):
        # calculate the difference of 2 float32 BGR images.
        d = (i1 - i2)
        d = d * d

        d = self._positive_sharpen(np.sum(d, -1), over_blur=over_blur)
        return d

    def _where_diff(self, img1=None, img2=None):
        if img1 is None:
            img1 = self.canvas
        if img2 is None:
            img2 = self.original_image

        # find out where max difference point is.
        d = self._diff(img1, img2, over_blur=True)

        i, j = np.unravel_index(d.argmax(), d.shape)
        return i, j, d

    def _get_phase(self, img):
        # grayify
        igray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32')

        # gradient
        xg = cv2.Sobel(igray, cv2.CV_32F, 1, 0, ksize=7)
        yg = - cv2.Sobel(igray, cv2.CV_32F, 0, 1, ksize=7)

        # in image axis y points downwards, hence the minus sign

        phase = cv2.phase(xg, yg)

        return phase  # radian

    # finally paint the same stroke onto the canvas.
    def _final_paint(self, color, brush, x, y, angle, radius, srad):
        self._compose(self.canvas, brush, x=x, y=y, rad=radius, srad=srad, angle=angle, color=color, usefloat=True,
                     useoil=True,
                     lock=self.canvas_lock)

    def _bgr2pwr(self, img):
        img = np.clip(img, a_max=1. - 1e-6, a_min=1e-6)
        img = np.power(img, 2.2 / self.bp_filter)
        return 1. - img

    def _pwr2bgr(self, img):
        img = 1. - img
        img = np.power(img, self.bp_filter / 2.2)
        return img

    # the brush process
    def _compose(self, canvas, brush, x, y, rad, srad, angle, color, usefloat=False, useoil=False, lock=None):
        """
        Put a brush on the canvas.

        :param canvas:
        :param brush:
        :param x:
        :param y:
        :param rad:
        :param srad:
        :param angle:
        :param color:
        :param usefloat:
        :param useoil:
        :param lock:
        :return:
        """
        # generate, scale and rotate the brush as needed
        brush_image = self.bf.rotate_brush(brush, rad, srad, angle)  # as alpha
        brush_image = np.reshape(brush_image, brush_image.shape + (1,))  # cast alpha into (h,w,1)

        # width and height of brush image
        bh = brush_image.shape[0]
        bw = brush_image.shape[1]

        y, x = int(y), int(x)

        # calculate roi params within orig to paint the brush
        ym, yp, xm, xp = y - bh / 2, y + bh / 2, x - bw / 2, x + bw / 2

        # w and h of orig
        orig_h, orig_w = canvas.shape[0:2]

        # crop the brush if exceed orig or <0
        alpha = brush_image[lc(0 - ym):lc(bh - (yp - orig_h)), lc(0 - xm):lc(bw - (xp - orig_w))]

        # crop the roi params if < 0
        ym, yp, xm, xp = lc(ym), lc(yp), lc(xm), lc(xp)

        if alpha.shape[0] == 0 or alpha.shape[1] == 0:  # or roi.shape[0]==0 or roi.shape[1]==0:
            # optimization: assume roi is valid
            raise ValueError('ROI should not be empty')

        # to simulate oil painting mixing:
        # color should blend in some fasion from given color to bg color
        if useoil:
            if not usefloat:  # to 0,1
                color = np.array(color).astype('float32') / 255.

            # gradient based color mixing
            # generate a blend map `gbmix` that blend in the direction of the brush stroke

            # first, generate xslope and yslope
            gby, gbx = np.mgrid[0:brush_image.shape[0], 0:brush_image.shape[1]]
            gbx = gbx / float(brush_image.shape[1]) - .5
            gby = gby / float(brush_image.shape[0]) - .5

            dgx, dgy = rn() - .5, rn() - .5  # variation to angle
            # then mix the slopes according to angle
            gbmix = gbx * math.cos(angle / 180. * math.pi + dgx) - gby * math.sin(angle / 180. * math.pi + dgx)

            # some noise?
            gbmix += np.random.normal(loc=0.15, scale=.2, size=gbmix.shape)

            # strenthen the slope
            gbmix = sigmoid_array(gbmix * 10)
            gbmix = np.reshape(gbmix, gbmix.shape + (1,)).astype('float32')

            gbmix = gbmix[lc(0 - ym):lc(bh - (yp - orig_h)), lc(0 - xm):lc(bw - (xp - orig_w))]

            alpha = alpha.astype('float32') / 255.

            # convert into oilpaint space
            color = self._bgr2pwr(color)

            def get_kernel_size(r):
                k = min(55, int(r))
                if k % 2 == 0:
                    k += 1
                if k < 3:
                    k += 2
                return k

            sdim = get_kernel_size(srad / 5)  # determine the blur kernel characteristics
            ldim = get_kernel_size(rad / 5)
            ssdim = get_kernel_size(srad / 7)

            # blur brush pattern
            softalpha = cv2.blur(alpha, (sdim, sdim))  # 0-1

            mixing_ratio = np.random.rand(alpha.shape[0], alpha.shape[1], 1)

            # increase mixing_ratio where brush pattern
            # density is lower than 1
            # i.e. edge enhance
            mixing_ratio[:, :, 0] += (1 - softalpha) * 2

            mixing_th = 0.1  # threshold, larger => mix more
            mixing_ratio = mixing_ratio > mixing_th
            # threshold into [0,1]

            # note: mixing_ratio is of dtype bool

            # apply motion blur on the mixed colormap
            kern = generate_motion_blur_kernel(dim=ldim, angle=angle)

            # sample randomly from roi
            # random is acturally bad idea
            # limit sample area under alpha is better
            ry, rx = 0, 0
            n = 20
            while n > 0:
                ry, rx = int(rn() * alpha.shape[0]), int(rn() * alpha.shape[1])
                if alpha[ry, rx] > .5:
                    break
                n -= 1

            # roi loading moved downwards, for optimization
            roi = canvas[ym:yp, xm:xp]

            if usefloat:  # roi to 0,1
                pass
            else:
                roi = roi.astype('float32') / 255.

            roi = self._bgr2pwr(roi)

            if n > 0:
                random_color = roi[ry, rx]
                tipcolor = color * gbmix + random_color * (1 - gbmix)
            else:
                tipcolor = color

            # blend tip color (image) with bg
            # larger the mixing_ratio, stronger the (tip) color
            ia = (1 - mixing_ratio).astype('float32')
            ca = tipcolor * mixing_ratio
            # print(roi.dtype,ia.dtype,ca.dtype)
            colormap = roi * ia + ca
            # print(colormap.dtype,kern.dtype)

            # mblur
            colormap = cv2.filter2D(colormap, cv2.CV_32F, kern)

            # final composition
            ca = colormap * alpha
            ia = 1 - alpha

            if lock is not None:
                lock.acquire()
                # print('lock acquired for brush @',x,y)
                # if canvas lock provided, acquire it. this prevents overwrite problems

            # final loading of roi.
            roi = canvas[ym:yp, xm:xp]

            if usefloat:
                roi = self._bgr2pwr(roi)
                canvas[ym:yp, xm:xp] = self._pwr2bgr(roi * ia + ca)
            else:
                roi = self._bgr2pwr(roi.astype('float32') / 255.)
                canvas[ym:yp, xm:xp] = self._pwr2bgr(roi * ia + ca) * 255.
        else:
            # no oil painting
            colormap = np.array(color).astype('float32')  # don't blend with bg, just paint fg

            if usefloat:
                alpha = alpha.astype('float32') / 255.
                ia = 1 - alpha
                ca = colormap * alpha
            else:
                colormap = colormap.astype('uint32')
                ia = 255 - alpha
                ca = colormap * alpha

            if lock is not None:
                lock.acquire()

            roi = canvas[ym:yp, xm:xp]

            if usefloat:
                canvas[ym:yp, xm:xp] = roi * ia + ca
            else:
                roi = roi.astype('uint32')
                canvas[ym:yp, xm:xp] = (roi * ia + ca) / 255

        # painted
        if lock is not None:
            lock.release()

    def _int_rad(self, orad, fatness):
        # obtain integer radius and shorter-radius
        radius = int(orad)
        srad = int(orad * fatness + 1)
        return radius, srad

    # get copy of square ROI area, to do drawing and calculate error.
    def _get_roi(self, newx, newy, radius):
        """
        Return copy of canvas and original image.

        :param newx:
        :param newy:
        :param radius:
        :return:
        """
        xshape = self.original_image.shape[1]
        yshape = self.original_image.shape[0]

        yp = int(min(newy + radius, yshape - 1))
        ym = int(max(0, newy - radius))
        xp = int(min(newx + radius, xshape - 1))
        xm = int(max(0, newx - radius))

        if yp <= ym or xp <= xm:
            # if zero w or h
            raise NameError('zero roi')

        original_roi = self.original_image[ym:yp, xm:xp]
        canvas_roi = self.canvas[ym:yp, xm:xp]
        canvas_roi = np.array(canvas_roi)

        return original_roi, canvas_roi

    # paint one stroke with given config and return the error.
    def _brush_try(self, color, brush, angle, nx, ny, radius, srad):
        """
        Try to put a brush on copy of the canvas, and return the error between this canvas and original image.

        :param color:
        :param brush:
        :param angle:
        :param nx:
        :param ny:
        :param radius:
        :param srad:
        :return:
        """
        original_roi, canvas_roi = self._get_roi(nx, ny, radius)

        self._compose(canvas_roi, brush, x=radius, y=radius, rad=radius, srad=srad, angle=angle, color=color,
                     usefloat=True, useoil=False)

        err_canvas = np.mean(self._diff(canvas_roi, original_roi))
        return err_canvas

    def _paint_one(self, x, y, brush_name='random', angle=-1., minrad=10, maxrad=60):
        """
        Paint one brush on canvas.
        If error decreases, we can draw on the real canvas in this way.
        Else, use graident decent to find a better place to put this brush.

        :param x:
        :param y:
        :param brush_name:
        :param angle:
        :param minrad:
        :param maxrad:

        :return: status: if the function successfully paint one brush
        """
        oradius = rn() * rn() * maxrad + minrad
        fatness = 1 / (1 + rn() * rn() * 6)
        max_try = 12
        min_try = 3
        brush, key = self.bf.get_brush(brush_name)

        # set initial angle
        if angle == -1.:
            angle = rn() * 360

        color = np.array(random.choice(self.target_color)).astype('float32')

        for i in range(max_try):
            try:
                radius, srad = self._int_rad(oradius, fatness)

                original_roi, canvas_roi = self._get_roi(x, y, radius)
                orig_err = np.mean(self._diff(canvas_roi, original_roi))

                # try to paint a brush on a new canvas
                err = self._brush_try(color, brush, angle, x, y, radius, srad)

                # if error decreased return results
                # else do gradient decent
                if err < orig_err and i >= min_try:
                    self._final_paint(color, brush, x, y, angle, radius, srad)
                    return True

            except ValueError as e:
                print('Error in calc_gradient: %s' % e)
                return False

            # do descend
            if i < max_try - 1:
                b, g, r = color[0], color[1], color[2]

                # calculate gradient
                err_canvas = self._brush_try((b + self.color_delta, g, r), brush, angle, x, y, radius, srad)
                db = err_canvas - err

                err_canvas = self._brush_try((b, g + self.color_delta, r), brush, angle, x, y, radius, srad)
                dg = err_canvas - err

                err_canvas = self._brush_try((b, g, r + self.color_delta), brush, angle, x, y, radius, srad)
                dr = err_canvas - err

                err_canvas = self._brush_try(color, brush, (angle + self.angle_delta) % 360, x, y, radius, srad)
                da = err_canvas - err

                err_canvas = self._brush_try(color, brush, angle, x + self.x_delta, y, radius, srad)
                dx = err_canvas - err

                err_canvas = self._brush_try(color, brush, angle, x, y + self.y_delta, radius, srad)
                dy = err_canvas - err

                err_canvas = self._brush_try(color, brush, angle, x, y, radius + self.radius_delta,
                                             srad + self.radius_delta)
                dradius = err_canvas - err

                dc = np.array([db, dg, dr]) / self.color_delta
                da = da / self.angle_delta
                dx = dx / self.x_delta
                dy = dy / self.y_delta
                dradius = dradius / self.radius_delta

                color = color - (dc * .3).clip(max=0.3, min=-0.3)
                color = color.clip(max=1., min=0.)
                angle = (angle - limit(da * 100000, -5, 5)) % 360
                x = x - limit(dx * 1000 * radius, -3, 3)
                y = y - limit(dy * 1000 * radius, -3, 3)
                oradius = oradius * (1 - limit(dradius * 20000, -0.2, .2))
                oradius = limit(oradius, 7, 100)

        return False

    def _put_strokes(self, batch_size=64, multi_thread=True):
        """
        Put `batch_size` strokes on canvas.
        Firstly, we need to sample some points where we can put a stroke.
        Then use _paint_one function to put this stroke properly.

        :param batch_size:
        :param multi_thread:
        :return:
        """
        points = []

        y, x, d = self._where_diff()
        phase_map = self._get_phase(self.original_image)

        # while not enough points:
        while len(points) < batch_size:
            # randomly pick one point
            yshape, xshape = self.original_image.shape[0:2]
            ry, rx = int(rn() * yshape), int(rn() * xshape)

            # accept with high probability if error is large
            # and vice versa
            # TODO: Now only draw the pixels which is original to be black
            if d[ry, rx] > 0.5 * rn() and (
                    self.original_image[ry, rx, 0] + self.original_image[ry, rx, 1] + self.original_image[
                ry, rx, 2] < 0.5 * 3):
                # get gradient orientation info from phase map
                phase = phase_map[ry, rx]  # phase should be between [0,2pi)

                # choose direction perpendicular to gradient
                angle = (phase / math.pi * 180 + 90) % 360

                points.append((ry, rx, angle))

        def _wrapper(idx):
            tup = points[idx]
            y, x, angle = tup
            self._paint_one(x, y, brush_name='random', minrad=10, maxrad=50, angle=angle)  # num of epoch

        if multi_thread:
            idx_list = range(len(points))
            pool = tp.ThreadPool(8)
            reqs = tp.makeRequests(_wrapper, idx_list)
            [pool.putRequest(req) for req in reqs]
            pool.wait()
        else:
            for idx, item in enumerate(points):
                print('single threaded mode.', idx)
                y, x, angle = item
                self._paint_one(x, y, brush_name='random', minrad=10, maxrad=50, angle=angle)  # num of epoch

    def paint(self, epoch=1, batch_size=64, result_dir='test'):
        """Convert image to oil style, save the results of every epoch in result_dir and return final result.

        :param epoch:
        :param batch_size:
        :param result_dir:
        :return: final result, with value in [0, 255]
        """
        for i in range(epoch):
            self._put_strokes(batch_size=batch_size, multi_thread=True)
            if result_dir is not None:
                print('Epoch %d: saving to disk...' % (i + 1))
                cv2.imencode('.png', self.canvas * 255)[1].tofile(
                    os.path.join(result_dir, '%d.png' % i))

        return np.array(self.canvas * 255)
