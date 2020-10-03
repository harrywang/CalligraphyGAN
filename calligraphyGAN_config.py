class Config:
    def __init__(self):
        self.max_epochs = 100
        self.save_model_epochs = 1
        self.print_steps = 1
        self.save_images_epochs = 1
        self.batch_size = 4
        self.learning_rate_D = 1e-4
        self.learning_rate_G = 1e-3
        self.k = 1  # the number of step of learning D before learning G (Not used in this code)
        self.num_classes = 1000  # number of classes for Calligraphy(should be 100, but 12 folder disappear)
        self.num_examples_to_generate = 25
        self.noise_dim = 128
        self.class_embedding_dim = 128
        self.class_dim = 1000
        self.image_size = 224
        self.checkpoint_dir = './checkpoints'
        self.example_dir = './example'


config = Config()
