async function checkSimilarity() {

    const question1 =
        document.getElementById("question1").value.trim();

    const question2 =
        document.getElementById("question2").value.trim();

    const result =
        document.getElementById("result");

    const error =
        document.getElementById("error");

    const loading =
        document.getElementById("loading");

    const button =
        document.getElementById("checkButton");

    if (!question1 || !question2) {

        error.textContent =
            "Please enter both questions.";

        error.classList.remove("hidden");

        result.classList.add("hidden");

        return;
    }

    error.classList.add("hidden");

    result.classList.add("hidden");

    loading.classList.remove("hidden");

    button.disabled = true;

    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        question1:
                            question1,

                        question2:
                            question2
                    })
                }
            );

        const data =
            await response.json();

        if (!data.success) {

            throw new Error(
                data.error ||
                "Prediction failed."
            );
        }

        document.getElementById(
            "resultText"
        ).textContent =
            data.result;

        document.getElementById(
            "percentage"
        ).textContent =
            data.similarity_percentage +
            "%";

        document.getElementById(
            "confidence"
        ).textContent =
            data.similarity_percentage +
            "%";

        document.getElementById(
            "threshold"
        ).textContent =
            data.threshold +
            "%";

        result.classList.remove(
            "hidden"
        );

    }

    catch (err) {

        error.textContent =
            err.message;

        error.classList.remove(
            "hidden"
        );
    }

    finally {

        loading.classList.add(
            "hidden"
        );

        button.disabled = false;
    }
}