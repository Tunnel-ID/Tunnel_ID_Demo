import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import orbitalAnimation from "../assets/orbitalAnimation.gif";
import tunnelLogo from "../assets/logo.png";
import whitepaper from "../assets/Tunnel_ID_white_paper.pdf";
import "../styles/HeroSection.css";

const HeroSection = () => {
  const navigate = useNavigate();
  const [isOptionsOpen, setIsOptionsOpen] = useState(false);
  const [showPopup, setShowPopup] = useState(false);
  const dropdownRef = useRef(null);

  // Function to connect MetaMask
  const connectWallet = async () => {
    if (window.ethereum) {
      try {
        const accounts = await window.ethereum.request({
          method: "eth_requestAccounts",
        });
        const walletAddress = accounts[0];
        navigate("/trimel-id", { state: { walletAddress } }); // Redirect to Trimel ID Page with wallet address
      } catch (error) {
        console.error("MetaMask connection failed:", error);
      }
    } else {
      alert("MetaMask not detected. Please install MetaMask.");
    }
  };

  const handleGetYourIDClick = () => {
    setShowPopup(true); // Show MetaMask connect popup
  };

  const handleLoginClick = () => {
    navigate("/Select-login");
  };

  const toggleOptions = () => {
    setIsOptionsOpen(!isOptionsOpen);
  };

  const handleAboutClick = () => {
    navigate("/about");
  };
  const handleFAQsClick = () => {
    navigate("/faqs"); // Navigate to the FAQs page
  };
  

  const handleApplicationClick = () => {
    navigate("/application");
  };

  const handleEcoSystemClick = () => {
    navigate("/eco-system");
  };

  const handleWhitePaperDownload = () => {
    const link = document.createElement("a");
    link.href = whitepaper; // Uses imported PDF
    link.setAttribute("target", "_blank"); // Opens in a new tab
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOptionsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  return (
    <div className="hero-section">
      <header className="navbar">
        <div className="brand">
          <img src={tunnelLogo} alt="Tunnel ID Logo" className="brand-logo" />
          <h1 className="logo-name">TUNNEL ID</h1>
        </div>
        <div className="button-group">
          <button className="get-id-button" onClick={handleGetYourIDClick}>
            Get Your ID
          </button>
          <button className="login-button1" onClick={handleLoginClick}>
            Verify Your ID
          </button>
          <button className="options-button" onClick={toggleOptions}>
            <div className="options-icon">&#9776;</div> {/* Hamburger icon */}
          </button>
        </div>
      </header>

      {/* Full-page dropdown overlay */}
      <div className={`dropdown-overlay ${isOptionsOpen ? "open" : ""}`}>
        <div className="dropdown-content" ref={dropdownRef}>
          <div className="dropdown-item" onClick={handleAboutClick}>About</div>
          <div className="dropdown-item" onClick={handleApplicationClick}>Application</div>
          <div className="dropdown-item" onClick={handleEcoSystemClick}>Eco Systems</div>
          <div className="dropdown-item" onClick={handleWhitePaperDownload}>White Paper</div>
          <div className="dropdown-item" onClick={handleFAQsClick}>FAQs</div>
          <div className="dropdown-item">Blog</div>
          <button className="transparent-button">Get In Touch</button>
        </div>
      </div>

      <div className="hero-content">
        <div className="image-container">
          <img
            src={orbitalAnimation}
            alt="Orbital Animation"
            className="orbital-image"
          />
        </div>
        <div className="text-content">
          <h1 className="hero-title">
            Tunnel ID Network <br /> Enables  
            <h1 className="des"> Decentralized,</h1> <br /> Privacy Preserving <br /> 
            <h1 className="des1">Biometric</h1> Proof of Humanness
          </h1>
          <p className="hero-description">
            Empowering humanity in the age of AI,<br /> Tunnel ID Network revolutionizes digital trust <br /> by creating keys from your biometrics <br /> with zero storage.
          </p>
        </div>
      </div>

      {/* MetaMask Connect Popup */}
      {showPopup && (
        <div className="popup">
          <p>Connect to MetaMask</p>
          <button onClick={connectWallet}>Connect MetaMask</button>
          <button onClick={() => setShowPopup(false)}>Cancel</button>
        </div>
      )}
    </div>
  );
};

export default HeroSection;
