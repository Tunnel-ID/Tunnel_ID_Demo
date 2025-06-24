import React from "react";
import "../styles/AboutPage.css";
import logo from "../assets/logo.png";

const AboutPage = () => {
  return (
    <div className="about-page">
      <div className="header-section">
        <img src={logo} alt="" className="brand-logo1" />
        <h1 className="logo-name1">TUNNEL ID</h1>
      </div>

      <section className="intro-section">
        <div className="intro-container">
          <div className="text-content">
            <h2>Redefining Digital Identity</h2>
            <p>
              Tunnel ID Network is a sybil-resistant decentralized identity protocol
              that empowers users to prove humanness and secure their digital life
              using a wide range of biometric modalities. It converts a user's unique
              biological traits into robust public/private key pairs without ever
              storing raw biometric data in any form.
            </p>
            <p>
              It leverages lattice-based cryptography and zero-knowledge proofs to
              deliver Proof of Humanness with dynamic fuzzy signatures that adapt to
              minor biometric variations, guaranteeing one human equal one unique
              identity and empowering users to securely authenticate, exchange
              digital assets, and access decentralized applications with unparalleled
              confidence and ease.
            </p>
          </div>
        </div>
      </section>

      <section className="benefits-section">
        <h2 className="key">Key Benefits</h2>
        <div className="benefits-grid">
          <div className="benefit-card">Inclusive and Accessible</div>
          <div className="benefit-card">Zero Biometric Footprint</div>
          <div className="benefit-card">Account Recovery and Auto Expiry</div>
          <div className="benefit-card">Robust Deduplication</div>
          <div className="benefit-card">Privacy Preserving On-chain verification</div>
          <div className="benefit-card">Unlinkability</div>
        </div>
      </section>

      <section className="meaning-section">
        <h2>What it means for you</h2>
        <p>
          With Tunnel ID Network, you can effortlessly unlock your digital life with
          your preferred biometric to create a secure, decentralized identity that's all
          yours. Log into apps, cast votes, and manage transactions with confidence,
          knowing your credentials are private, untraceable, and always under your
          control.
        </p>
        <p>
          Enjoy a seamless, worry-free online experience where your identity evolves
          with you, remaining secure and accessible every step of the way.
        </p>
        <div className="try-it-out-section">
          <a href="/" className="try-it-out-link">
            ➡️ Try it out
          </a>
        </div>
      </section>

      <section className="newsletter-section">
        <h2>Subscribe to Tunnel ID Newsletter</h2>
        <p>
          Get exclusive insights about how we create a secure and inclusive
          decentralized identity that’s all yours.
        </p>
        <form className="newsletter-form">
          <input type="email" placeholder="Your Email" required />
          <button type="submit">Subscribe</button>
        </form>
      </section>
    </div>
  );
};

export default AboutPage;