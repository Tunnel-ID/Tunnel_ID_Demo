import React, { useState } from "react";
import "../styles/FacialCredentialForm.css"; // Import the corresponding CSS

const FacialCredentialForm = () => {
  const [userId, setUserId] = useState("");
  const [emailId, setEmailId] = useState("");
  const [message, setMessage] = useState("");

  // ✅ Set correct backend URL
  const BACKEND_URL = "https://tunnelid.me/api"; // Recommended (Use Domain)
  // const BACKEND_URL = "http://3.6.137.82/api"; // Alternative (Use Public IP)

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      // ✅ Updated Fetch URL for EC2
      const response = await fetch(`${BACKEND_URL}/register_facial`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          userId,
          emailId,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessage(data.message); // Success message
      } else {
        setMessage(data.message); // Error message
      }
    } catch (error) {
      setMessage("An error occurred while registering.");
      console.error("Error:", error);
    }
  };

  return (
    <div className="facial-credential-form">
      <h2>Create Facial Credential</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="userId">User ID:</label>
          <input
            type="text"
            id="userId"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="emailId">Email ID:</label>
          <input
            type="email"
            id="emailId"
            value={emailId}
            onChange={(e) => setEmailId(e.target.value)}
            required
          />
        </div>
        <button type="submit">Submit</button>
      </form>
      {message && <p className="message">{message}</p>}
    </div>
  );
};

export default FacialCredentialForm;
