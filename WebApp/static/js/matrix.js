const canvas = document.getElementById('matrixRain');
const ctx = canvas.getContext('2d');

canvas.height = window.innerHeight;
canvas.width = window.innerWidth;

let matrix = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789@#$%^&*()*&^%";
matrix = matrix.split("");

const fontSize = 14;
const columns = canvas.width / fontSize;

const drops = [];

for (let x = 0; x < columns; x++) drops[x] = 1;

function draw() {
    ctx.fillStyle = "rgba(0, 0, 0, 0.08)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "#00ff41";
    ctx.font = fontSize + "px monospace";

    for (let i = 0; i < drops.length; i++) {
        const text = matrix[Math.floor(Math.random() * matrix.length)];
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);
        
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.95)
            drops[i] = 0;

        drops[i]++;
    }
}

setInterval(draw, 33);
