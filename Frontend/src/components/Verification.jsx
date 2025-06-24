import { useState, useEffect } from "react";
import "../styles/Verification.css";

export default function Verification() {
  const [id, setId] = useState("");
  const [message, setMessage] = useState("");
  const [messageClass, setMessageClass] = useState("");
  const [sketchHex, setSketchHex] = useState(null);

  // ✅ Set correct backend URL
  const BACKEND_URL = "https://tunnelid.me/api"; // Recommended (Domain)
  // const BACKEND_URL = "http://3.6.137.82/api"; // Alternative (Use Public IP)

  useEffect(() => {
    fetch(`${BACKEND_URL}/get_sketch_hex`)
      .then((response) => response.json())
      .then((data) => {
        if (data.sketch_hex) {
          console.log("Fetched sketch_hex:", data.sketch_hex);
          setSketchHex(data.sketch_hex);
        } else {
          console.error("No sketch_hex found");
        }
      })
      .catch((error) => console.error("Error fetching sketch_hex:", error));
  }, []);

  const handleVerify = async () => {
    if (id.trim() === "") {
      setMessage("⚠️ Please enter a valid Base64 sketch.");
      setMessageClass("failure");
      return;
    }

    try {
      // ✅ Updated Fetch URL for EC2
      const response = await fetch(`${BACKEND_URL}/verify_fingerprint`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sketch_b64: id.trim() }), // Send Base64 instead of sketch_hex
      });

      const data = await response.json();
      setMessage(data.message);
      setMessageClass(data.status === "success" ? "success" : "failure");
    } catch (error) {
      console.error("Error verifying:", error);
      setMessage("⚠️ Server error! Please try again.");
      setMessageClass("failure");
    }
  };

  return (
    <div className="verification-container">
      <h2>🔒 Secure Verification</h2>
      <textarea
        className="id-input"
        placeholder="Enter your ID here..."
        value={id}
        onChange={(e) => setId(e.target.value)}
      ></textarea>
      <button className="verify-btn" onClick={handleVerify}>
        Verify
      </button>
      {message && <p className={`message ${messageClass}`}>{message}</p>}
    </div>
  );
}
