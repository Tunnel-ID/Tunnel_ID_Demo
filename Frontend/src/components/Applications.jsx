import React, { useState, useEffect, useCallback } from "react";
import "../styles/Applications.css";
import image1 from "../assets/image1.gif";
import image4 from "../assets/image4.gif";
import image2 from "../assets/image2.gif";
import image5 from "../assets/image5.gif";
import image3 from "../assets/image3.gif";
import logo from "../assets/logo.png";

const TunnelID = () => {
  const [activeSection, setActiveSection] = useState(0);
  const [isScrolling, setIsScrolling] = useState(false);
  const [scrollDirection, setScrollDirection] = useState("down");

  const sections = [
    {
      title: "Decentralized Finance (DeFi) and KYC",
      text: "Access DeFi platforms with confidence knowing that every participant is a verified, unique human. Say goodbye to bots and fraudulent activity, and experience seamless, secure transactions without the hassle of traditional KYC.",
      gif: image1,
    },
    {
      title: "Empower DAO Governance & Fair Voting",
      text: "With Tunnel ID, it's one human, one vote—ensuring truly democratic governance and eliminating Sybil attacks. Cast your vote and shape decentralized communities with the assurance that your participation is backed by unforgeable biometric proof.",
      gif: image2,
    },
    {
      title: "Seamless Secure Access to dApps",
      text: "Enjoy frictionless logins across decentralized applications. Whether it's social platforms, gaming, or smart contracts, your Tunnel ID provides instant, secure access without exposing your raw biometric data—your private keys stay exclusively in your control.",
      gif: image3,
    },
    {
      title: "Revolutionize NFT Marketplaces & Digital Art",
      text: "Join a vibrant ecosystem where authenticity matters. Use your Tunnel ID to securely mint, buy, and sell NFTs knowing that your biometric credentials are privacy-preserving and completely unlinkable to your transactions, ensuring a fraud-free environment.",
      gif: image4,
    },
    {
      title: "One Key. All Your Assets",
      text: "Simplify your crypto experience by unifying all your wallets under one secure roof. With your biometric identity as the master key, manage diverse digital assets and engage in frictionless transactions.",
      gif: image5,
    },
  ];

  const handleScroll = useCallback(
    (direction) => {
      if (isScrolling) return;

      setIsScrolling(true);
      setScrollDirection(direction);

      setActiveSection((prev) =>
        direction === "down"
          ? Math.min(prev + 1, sections.length - 1)
          : Math.max(prev - 1, 0)
      );

      setTimeout(() => {
        setIsScrolling(false);
      }, 800); // Timeout matches animation duration
    },
    [isScrolling, sections.length]
  );

  useEffect(() => {
    const onScroll = (e) => {
      if (e.deltaY > 0) handleScroll("down");
      else if (e.deltaY < 0) handleScroll("up");
    };

    window.addEventListener("wheel", onScroll);
    return () => window.removeEventListener("wheel", onScroll);
  }, [handleScroll]);

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === "ArrowDown") handleScroll("down");
      else if (e.key === "ArrowUp") handleScroll("up");
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handleScroll]);

  return (
    <>
      <img src={logo} alt="Logo" className="logo5" />
      <h1 className="logo-name2">TUNNEL ID</h1>
      <div className="tunnel-id-container">
        {/* Main Content Section */}
        <div className="content">
          {/* Left Column for Text */}
          <div className="text-column">
            <h2>Applications</h2>
            {sections.map((section, index) => (
              <div
                key={index}
                className={`section ${
                  index === activeSection
                    ? scrollDirection === "down"
                      ? "slide-up"
                      : "slide-down"
                    : ""
                }`}
                style={{
                  display: index === activeSection ? "block" : "none",
                }}
              >
                <h3>{section.title}</h3>
                <p>{section.text}</p>
                <div className="divider"></div>
              </div>
            ))}
          </div>

          {/* Right Column for GIF */}
          <div className="gif-column">
            {sections.map((section, index) => (
              <div
                key={index}
                className={`gif-box ${
                  index === activeSection
                    ? scrollDirection === "down"
                      ? "slide-in-right"
                      : ""
                    : ""
                }`}
                style={{
                  display: index === activeSection ? "block" : "none",
                }}
              >
                <img
                  src={section.gif}
                  alt={`GIF ${index + 1}`}
                  className="gif"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Navigation Dots */}
        <div className="nav-dots">
          {sections.map((_, index) => (
            <span
              key={index}
              className={`dot ${index === activeSection ? "active" : ""}`}
              onClick={() => setActiveSection(index)}
            ></span>
          ))}
        </div>

        {/* Down Arrow */}
        {activeSection < sections.length - 1 && (
          <div className="down-arrow" onClick={() => handleScroll("down")}>
            <span>▼</span>
          </div>
        )}
      </div>
    </>
  );
};

export default TunnelID;