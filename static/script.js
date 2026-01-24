document.getElementById('churnForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Show loading state
    const btn = document.getElementById('predictBtn');
    const originalText = btn.innerText;
    btn.innerText = "Analyzing...";
    btn.disabled = true;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        btn.innerText = originalText;
        btn.disabled = false;

        if (result.error) {
            alert("Error: " + result.error);
        } else {
            showResult(result);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        btn.innerText = originalText;
        btn.disabled = false;
        alert("An error occurred. Please check the console for details.");
    });
});

function showResult(data) {
    const container = document.getElementById('resultContainer');
    const text = document.getElementById('resultText');

    container.classList.remove('hidden');

    let churnHtml = data.churn === "Yes"
        ? `<span class="churn-yes">High Risk of Churn</span>`
        : `<span class="churn-no">Low Risk of Churn</span>`;

    text.innerHTML = `Prediction: ${churnHtml} <br> Churn Probability: <strong>${data.probability}%</strong>`;

    // Smooth scroll to the result
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeResult() {
    document.getElementById('resultContainer').classList.add('hidden');
    document.getElementById('churnForm').reset();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}