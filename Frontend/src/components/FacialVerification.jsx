import React, { useState } from "react";
import "../styles/FacialVerification.css"; // Import CSS
import { motion } from "framer-motion"; // For animations

export default function FacialVerification() {
  const [sketchId, setSketchId] = useState("");
  const [message, setMessage] = useState("");
  const [messageClass, setMessageClass] = useState("");

  // ✅ Set correct backend URL
  const BACKEND_URL = "https://tunnelid.me/api"; // Recommended (Use Domain)
  // const BACKEND_URL = "http://3.6.137.82/api"; // Alternative (Use Public IP)

  const handleVerify = async () => {
    if (sketchId.trim() === "") {
      setMessage("⚠️ Please enter a valid Sketch ID.");
      setMessageClass("failure");
      return;
    }

    try {
      // ✅ Updated Fetch URL for EC2
      const response = await fetch(`${BACKEND_URL}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sketch_base64: sketchId.trim() }),
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
    <motion.div 
      className="facial-verification-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1 }}
    >
      <h2 className="facial-title">🔍 Sketch ID Verification</h2>

      <motion.input
        type="text"
        className="facial-input"
        placeholder="Enter Sketch ID (Base64)..."
        value={sketchId}
        onChange={(e) => setSketchId(e.target.value)}
        whileFocus={{ scale: 1.05 }}
      />

      <motion.button 
        className="verify-btn" 
        onClick={handleVerify}
        whileTap={{ scale: 0.9 }}
      >
        ✅ Verify
      </motion.button>

      {message && <p className={`message ${messageClass}`}>{message}</p>}
    </motion.div>
  );
}
