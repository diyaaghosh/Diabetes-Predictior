document.getElementById("predictionForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const data = {
        Pregnancies: Number(document.getElementById("pregnancies").value),
        Glucose: Number(document.getElementById("glucose").value),
        BloodPressure: Number(document.getElementById("bloodPressure").value),
        SkinThickness: Number(document.getElementById("skinThickness").value),
        Insulin: Number(document.getElementById("insulin").value),
        BMI: Number(document.getElementById("bmi").value),
        DiabetesPedigreeFunction: Number(document.getElementById("dpf").value),
        Age: Number(document.getElementById("age").value)
    };

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    const probability = result.probability;

    document.getElementById("result").style.display = "block";
    document.getElementById("progressBar").style.width =
        (probability * 100).toFixed(2) + "%";

    document.getElementById("probability").innerHTML =
        `<b>Estimated Probability: ${(probability * 100).toFixed(2)}%</b>`;

    if (probability < 0.30) {
        document.getElementById("status").innerHTML =
            `<div class="success">🟢 Low Diabetes Risk (${(probability * 100).toFixed(2)}%)</div>`;
    } else {
        document.getElementById("status").innerHTML =
            `<div class="error">🔴 High Diabetes Risk (${(probability * 100).toFixed(2)}%)</div>`;
    }
});