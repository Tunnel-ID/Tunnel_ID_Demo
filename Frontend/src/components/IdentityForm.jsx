import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/IdentityForm.css";

const IdentityForm = () => {
  const [fingerprint, setFingerprint] = useState(null);
  const navigate = useNavigate();

  // ✅ Use correct backend URL
  const BACKEND_URL = "https://tunnelid.me/api"; // Recommended (Use Domain)
  // const BACKEND_URL = "http://3.6.137.82/api"; // Alternative (Use Public IP)

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!fingerprint) {
      alert("Please upload a fingerprint file.");
      return;
    }

    const formData = new FormData();
    formData.append("fingerprint_file", fingerprint);

    try {
      // ✅ Updated Fetch Request to Work with Backend
      const response = await fetch(`${BACKEND_URL}/register_fingerprint`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        console.log("Registration successful:", result);
        alert("Fingerprint registered successfully!");
        navigate("/trimel-id"); // ✅ Redirect after success
      } else {
        console.error("Error from backend:", result);
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error("Network or Server Error:", error);
      alert("Server error! Please try again.");
    }
  };

  return (
    <div className="identity-form-container">
      <div className="blockchain-animation">
        {[...Array(20)].map((_, index) => (
          <div key={index} className="block-node" />
        ))}
      </div>

      <h2 className="form-title">Upload Your Fingerprint</h2>
      <form onSubmit={handleSubmit} className="identity-form">
        <label>Fingerprint File:</label>
        <input
          type="file"
          onChange={(e) => setFingerprint(e.target.files[0])}
          accept=".npy"
          required
        />

        <button type="submit" className="submit-button">
          Submit
        </button>
      </form>
    </div>
  );
};

export default IdentityForm;
