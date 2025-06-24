import React from 'react';
import '../styles/Ecosystem.css';
import logo from '../assets/logo.png'; // Assuming your logo is in the assets folder

const EcoSystem = () => {
  return (
    <div className="tunnel-id-page-1">
      <div className="tunnel-id-container-1">
        {/* Logo and Tunnel ID Title */}
        <div className="lago-section">
          <img src={logo} alt="Tunnel ID Logo" className="lago" />
          <h1 className="tunnel-id-titl">Tunnel ID</h1>
        </div>

        {/* Eco System section */}
        <section className="eco-system-section">
          <h2 className="eco-system-title">Eco System</h2>

          {/* Tunnel ID Protocol section */}
          <article className="tunnel-id-protocol-section-1">
            <h3 className="tunnel-id-protocol-title-1">Tunnel ID</h3>
            <p className="tunnel-id-protocol-content-1">
              At the heart of our ecosystem lies the Tunnel ID Protocol, a decentralized proof of perceived solution. It harnesses advanced biometric cryptography to transform any biometric modality into unforgeable public/private cryptographic key parts without ever storing any sensitive data. Using techniques such as fuzzy signature generation, we can produce knowledge products and manipulate on-chain verification via Merkle Trees. Tunnel ID ensures that each human identity is unique, valid, resistant, and quantum secure. This protocol lays the foundation for trust in the digital realm by ensuring one human equals one verified identity.
            </p>
          </article>

          {/* Tunnel Wallet section */}
          <article className="tunnel-wallet-section-1">
            <h3 className="tunnel-wallet-title-1">Tunnel Wallet</h3>
            <p className="tunnel-wallet-content-1">
              The Tunnel Wallet is your portal into the entire Tunnel Network ecosystem as well as the managing digital assets in the Web3 world. Not only does it enable us to execute seamless transactions on the Tunnel Network, but it also consolidates all the platforms where you've used your Tunnel ID for verification. With advanced features like key refresh, your public/private key parts are continuously updated to maintain optimal security, ensuring that your biometric-based identity remains robust and uncompromised. Timelines complete correct, transparency, and peace of mind as Tunnel Wallet empowers you to manage your digital footprint effortlessly while transacting with confidence across the decentralized web.
            </p>
          </article>

          {/* Tunnel Network section */}
          <article className="tunnel-network-section-1">
            <h3 className="tunnel-network-title-1">Tunnel Network</h3>
            <p className="tunnel-network-content-1">
              The Tunnel Network bridges the gap between secure identity verification and the expansive world of Web3 applications. Acting as the connective tissue of our ecosystem, the Tunnel Network integrates the Tunnel ID Protocol and Tunnel Wallet to power a diverse range of decentralized services, from Defi Publications and DAO governance to NFT management and beyond. It ensures that every interaction, whether it's wiring in a decentralized community or truncating access to multiple chunks, is backed by a verifiable, privacy-preserving identity. This unified network paves the way for a truly human-centric digital economy. With our advanced biometric cryptography, zero-knowledge proofs, and quantum-resistant security, it offers you a robust, privacy-preserving foundation for verifying true human identity. Driving dApps that not only safeguard user interactions but also ensure every transaction and engagement is powered by genuine unforgeable proof of humanness.
            </p>
          </article>
        </section>
      </div>
    </div>
  );
};

export default EcoSystem;
