const backend_url = "https://procid-ocr-rrecognition-backend.hf.space/api/predict"

document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("sub");
    btn.addEventListener("click", processOCR);

    
});

async function processOCR(e) {
        e.preventDefault(); 
        const fileInput = document.getElementById("imageUpload");
        const output = document.getElementById("output");

        if (fileInput.files.length === 0) {
            alert("Please upload image first, ok?");
            return;
        }

        const formData = new FormData();
        formData.append('aiImage', fileInput.files[0]);
        try {
            btn.disabled = true;
            output.textContent = "...";

            const Response = await fetch(backend_url, {
                method: 'POST',
                body: formData
            })

            if (!Response.ok) {
                throw new Error(`Sorry, server returned with status code ${Response.status}`);
            }

            const result = await Response.json();
            output.textContent = `${result.prediction}`;


        } catch (error) {
            console.error("Connection Error: ", error);
            output.textContent=("Sorry, unable to send, error.");
        } finally {
            btn.disabled = false;
        }
    }