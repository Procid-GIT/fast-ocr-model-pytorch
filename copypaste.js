const cbutton = document.getElementById("copy");
const output = document.getElementById("output");

cbutton.addEventListener('click', copytoClipboard);

async function copytoClipboard() {
    const intext = output.innerText;
    const txt = intext.slice(50, 51);

    try{
        await navigator.clipboard.writeText(txt);
        const origin = cbutton.innerText;
        cbutton.innerText = "Copied To Clipboard!";
        cbutton.disabled = true;
        
        setTimeout(() => {
            cbutton.innerText = origin;
            cbutton.disabled = false;
        }, 2000);

    } catch (err) {
        cbutton.innerText = "Failed to Copy!";
        setTimeout(() => {
            cbutton.innerText = origin;
            cbutton.disabled = false;
        }, 2000);
    }
}
