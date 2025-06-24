import React from "react";
import { UserCircle } from "lucide-react";
import "../styles/UserIdPage.css";
import logo from "../assets/logo.png";

const UserIdPage = () => {
  // ✅ Use the correct domain dynamically
  const DOMAIN = "https://tunnelid.me"; // Change to Public IP if needed
  const userId = "0xA568JQ"; // Sample User ID for demonstration
  const userLink = `${DOMAIN}/${userId}`;

  return (
    <div className="container">
      {/* Logo Section */}
      <header className="header">
        <div className="logo-section">
          <img src={logo} alt="Tunnel ID Logo" className="logo-img" />
          <h1 className="logo-text">Tunnel ID</h1>
        </div>
      </header>

      {/* Profile Section */}
      <div className="profile">
        <div className="avatar">
          <UserCircle size={80} />
        </div>
        <p className="user-id">ID: <a href={userLink} target="_blank" rel="noopener noreferrer">{userLink}</a></p>
      </div>

      {/* Welcome Message */}
      <h1 className="welcome-message">Welcome to Tunnel ID, You are the key</h1>

      {/* Identity Section */}
      <h2 className="identity-section">Access Your Credentials</h2>

      {/* Credential Buttons */}
      <div className="credentials-grid">
        {["Fingerprint", "Facial", "Iris", "Voice"].map((title, index) => (
          <div key={index} className="credential-button">
            {title} Credential
          </div>
        ))}
      </div>
    </div>
  );
};

export default UserIdPage;
