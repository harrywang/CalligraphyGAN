cards = document.getElementsByClassName("card");
console.log(cards);

Array.prototype.forEach.call(cards, card => {
    card.onclick = chooseStyle;
});

function chooseStyle() {
    document.getElementById("style_text").value = this.innerText;
    cards = document.getElementsByClassName("card");
    Array.prototype.forEach.call(cards, card => {
        card.classList.value = "card";
    });

    this.classList.value = "card h-100 text-white bg-primary";
}