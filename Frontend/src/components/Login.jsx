import React, { useState } from "react";
import { useNavigate } from "react-router-dom"; // Import useNavigate
import "../styles/Login.css";

const Login = () => {
  const [userId, setUserId] = useState("");
  const [fingerprint, setFingerprint] = useState(null);
  const [message, setMessage] = useState("");
  const navigate = useNavigate(); // Initialize useNavigate

  // ✅ Use the correct backend URL (EC2 IP or Domain)
  const BACKEND_URL = "https://tunnelid.me/api";  // Recommended (Use Domain)
  // const BACKEND_URL = "http://3.6.137.82/api"; // Alternative (Use Public IP)

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!userId || !fingerprint) {
      setMessage("Please enter User ID and upload a fingerprint.");
      return;
    }

    // Create a FormData object to send the data
    const formData = new FormData();
    formData.append("user_id", userId);
    formData.append("fingerprint_file", fingerprint);

    try {
      // ✅ Updated Fetch Request to Work with Backend
      const response = await fetch(`${BACKEND_URL}/login`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setMessage("Login successful!");
        navigate("/dashboard"); // ✅ Redirect to dashboard on success
      } else {
        setMessage(`Login failed: ${result.error}`);
      }
    } catch (error) {
      setMessage("An error occurred during login. Please try again.");
      console.error("Error during login:", error);
    }
  };

  return (
    <div className="login-container">
      <h2 className="login-title">Login to Your Account</h2>
      <form onSubmit={handleSubmit} className="login-form">
        {/* User ID Input */}
        <label htmlFor="userId">User ID:</label>
        <input
          type="text"
          id="userId"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          placeholder="Enter your User ID"
          required
        />

        {/* Fingerprint Input */}
        <label htmlFor="fingerprint">Fingerprint Authentication:</label>
        <input
          type="file"
          id="fingerprint"
          accept=".npy" // Ensure only .npy files are accepted
          onChange={(e) => setFingerprint(e.target.files[0])}
          required
        />

        {/* Submit Button */}
        <button type="submit" className="login-button">Login</button>
      </form>

      {/* Display login status message */}
      {message && <p className="login-message">{message}</p>}
    </div>
  );
};

export default Login;
