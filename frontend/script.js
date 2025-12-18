async function uploadPDF() {
  const fileInput = document.getElementById("pdfInput");
  if (!fileInput.files[0]) {
    alert("Please select a PDF first!");
    return;
  }

  const formData = new FormData();
  formData.append("pdf", fileInput.files[0]);

  const res = await fetch("http://localhost:8000/upload", {
    method: "POST",
    body: formData
  });
  const result = await res.json();
  alert(result.message || result.error);
}

async function askQuestion() {
  const query = document.getElementById("query").value.trim();
  const answerBox = document.getElementById("answerText");

  if (!query) {
    alert("Enter a question!");
    return;
  }

  answerBox.textContent = "⏳ Thinking...";
  const res = await fetch("http://localhost:8000/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query })
  });

  const result = await res.json();
  if (result.answer) {
    answerBox.textContent = result.answer;
  } else {
    answerBox.textContent = "⚠️ Error: " + result.error;
  }
}
