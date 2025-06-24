import React, { useState } from "react";
import "../styles/Footer.css"; // Import the CSS file
import { FaTwitter, FaDiscord, FaTelegram, FaLinkedin, FaMedium } from "react-icons/fa"; // Import icons from react-icons
import { useNavigate } from "react-router-dom";

const Footer = () => {
  const [showPopup, setShowPopup] = useState(false); // State to control popup visibility
  const navigate = useNavigate();

  // Function to handle the "Terms & Conditions" click
  const handleTermsClick = (event) => {
    event.preventDefault(); // Prevent default behavior (e.g., navigation or download)
    setShowPopup(true); // Show the popup
  };

  // Function to close the popup
  const closePopup = () => {
    setShowPopup(false); // Hide the popup
  };

  // Function to handle "Read More" click
  const handleReadMore = () => {
    navigate("/terms-and-conditions");
  };

  return (
    <footer className="footer">
      <div className="social-icons">
        <a href="medium.com/@tunnelid" target="_blank" rel="noopener noreferrer">
          <FaMedium className="icon" />
        </a>
        <a href="https://x.com/Tunnel_Id" target="_blank" rel="noopener noreferrer">
          <FaTwitter className="icon" />
        </a>
        <a href="https://t.co/eGMmi3xKGS" target="_blank" rel="noopener noreferrer">
          <FaDiscord className="icon" />
        </a>
        <a href="https://www.linkedin.com/company/tunnelid/" target="_blank" rel="noopener noreferrer">
          <FaLinkedin className="icon" />
        </a>
        <a href=" https://t.me/Tunnelid01" target="_blank" rel="noopener noreferrer">
          <FaTelegram className="icon" />
        </a>
      </div>
      <div className="footer-links">
        <a
          href="/terms-and-conditions"
          className="terms-link"
          onClick={handleTermsClick}
        >
          Terms & Conditions
        </a>
      </div>

      {/* Popup for Terms & Conditions */}
      {showPopup && (
        <div className="popup-overlay">
          <div className="popup-content">
            <h2>Terms & Conditions</h2>
            <p>
              Tunnel ID Network's Terms & Conditions outline user eligibility, account security, biometric data processing, and permissible use. Users retain control over private keys, while transactions occur on decentralized networks. Tunnel ID Network disclaims warranties, limits liability, and reserves the right to modify terms or terminate services.
            </p>
            <button className="read-more-button" onClick={handleReadMore}>
              Read More
            </button>
            <button className="close-button" onClick={closePopup}>
              Close
            </button>
          </div>
        </div>
      )}
    </footer>
  );
};

export default Footer;
