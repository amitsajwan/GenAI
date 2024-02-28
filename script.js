// Request data goes here
// The example below assumes JSON formatting which may be updated
// depending on the format your endpoint expects.
// More information can be found here:
// https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
const requestBody = "Hello I am finistra Defect Chatbot" ;

const requestHeaders = new Headers({"Content-Type" : "application/json"});

// Replace this with the primary/secondary key or AMLToken for the endpoint
const apiKey = "noOjCLypMfKQQTwh8JuJU98vBLpheZuQ"; 
if (!apiKey)
{
	throw new Exception("A key should be provided to invoke the endpoint");
}
requestHeaders.append("Authorization", "Bearer " + apiKey)

// This header will force the request to go to a specific deployment.
// Remove this line to have the request observe the endpoint traffic rules
requestHeaders.append("azureml-model-deployment", "amitsajwan-project-1-btrga-1");

const url = "https://api-defect.azure-api.net/defects/";


fetch(url, {
  method: "POST",
  body: JSON.stringify(requestBody),
  headers: requestHeaders,
  mode: "no-cors"
})
	.then((response) => {
	if (response.ok) {
		return response.json();
	} else {
		// Print the headers - they include the request ID and the timestamp, which are useful for debugging the failure
		console.debug(...response.headers);
		console.debug(response.body)
		throw new Error("Request failed with status code" + response.status);
	}
	})
	.then((json) => console.log(json))
	.catch((error) => {
		console.error(error)
	});
document.getElementById("sendRequest").addEventListener("click", function() {
    const userInput = document.getElementById("textInput").value;

    const requestBody = { "text": userInput }; // Modify as needed based on your model input format

    // Your existing code for making the fetch request
    // Make sure to replace the requestBody with the user input
    fetch(url, {
        method: "POST",
        body: JSON.stringify(requestBody),
        headers: requestHeaders
    })
    .then((response) => {
        if (response.ok) {
            return response.json();
        } else {
            console.debug(...response.headers);
            console.debug(response.body)
            throw new Error("Request failed with status code" + response.status);
        }
    })
    .then((json) => {
        // Display the response to the user
        document.getElementById("response").innerText = JSON.stringify(json, null, 2);
    })
    .catch((error) => {
        console.error(error)
    });
});
