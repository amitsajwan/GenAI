const url = "https://api-defect.azure-api.net/defects/";

const requestBody = {
  "text": "Hello I am finistra Defect Chatbot"
}; // Modify as needed based on your model input format

const requestHeaders = new Headers({
  "Content-Type": "application/json",
  "Authorization": "Bearer " + apiKey,
  "azureml-model-deployment": "amitsajwan-project-1-btrga-1"
});

document.getElementById("sendRequest").addEventListener("click", function () {
  const userInput = document.getElementById("textInput").value;
  const requestBody = { "text": userInput }; // Modify as needed based on your model input format

  fetch(url, {
      method: "POST",
      body: JSON.stringify(requestBody),
      headers: requestHeaders,
      mode: "no-cors" // Setting no-cors mode to bypass CORS restrictions
    })
    .then((response) => {
      if (response.ok) {
        return response.json();
      } else {
        console.debug(...response.headers);
        console.debug(response.body);
        throw new Error("Request failed with status code" + response.status);
      }
    })
    .then((json) => {
      // Display the response to the user
      document.getElementById("response").innerText = JSON.stringify(json, null, 2);
    })
    .catch((error) => {
      console.error(error);
    });
});
