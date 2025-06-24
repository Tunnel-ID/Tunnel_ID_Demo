import React, { useState } from "react";
import "../styles/FAQS.css"; // Import the CSS file

const faqsData = [
  {
    category: "Tunnel ID - Frequently Asked Questions (FAQ)",
    items: [
      {
        question: "What is Tunnel ID?",
        answer:
          "Tunnel ID is a privacy-preserving, decentralized biometric identity protocol that allows users to verify their identity without storing raw biometric data.",
      },
      {
        question: "How is Tunnel ID different from traditional biometric systems?",
        answer:
          "Unlike traditional systems that store biometrics, Tunnel ID converts biometric data into cryptographic proofs, ensuring privacy and security.",
      },
      {
        question: "Why should I use Tunnel ID?",
        answer:
          "Tunnel ID offers self-sovereign identity, wallet unification, Sybil resistance, and privacy-first authentication while preventing identity theft and fraud.",
      },
    ],
  },
  {
    category: "Privacy & Security",
    items: [
      {
        question: "Does Tunnel ID store my biometric data?",
        answer:
          "No, Tunnel ID never stores raw biometric data. It only uses cryptographic representations that cannot be reversed to reconstruct biometric features.",
      },
      {
        question: "How does Tunnel ID ensure privacy?",
        answer:
          "It leverages zero-knowledge proofs (ZKPs) and lattice-based cryptography to allow identity verification without revealing sensitive biometric details.",
      },
      {
        question: "Is Tunnel ID resistant to quantum computing attacks?",
        answer:
          "Yes, Tunnel ID uses post-quantum cryptographic techniques, making it secure against future quantum threats.",
      },
    ],
  },
  {
    category: "Wallet & Blockchain Integration",
    items: [
      {
        question: "Can I use Tunnel ID with multiple blockchain wallets?",
        answer:
          "Yes, Tunnel ID unifies wallet addresses across different blockchains, ensuring seamless identity management.",
      },
      {
        question: "Does Tunnel ID work with DeFi and DAOs?",
        answer:
          "Yes, Tunnel ID enhances trust in DeFi and DAO governance by ensuring that only verified human users participate.",
      },
    ],
  },
  {
    category: "Account Recovery & Refresh",
    items: [
      {
        question: "What happens if I lose access to my device?",
        answer:
          "Tunnel ID supports biometric-based recovery through secondary biometrics, allowing you to regain access without compromising security.",
      },
      {
        question: "How does Tunnel ID prevent long-term biometric theft?",
        answer:
          "Tunnel ID uses an automatic refresh mechanism, requiring users to periodically re-verify to maintain security.",
      },
    ],
  },
  {
    category: "Using Tunnel ID",
    items: [
      {
        question: "How do I sign up for Tunnel ID?",
        answer:
          "Simply scan your biometric data through a Tunnel ID-compatible application, and a cryptographic proof will be generated without storing your raw biometrics.",
      },
      {
        question: "Can I delete my Tunnel ID?",
        answer:
          "Since no biometric data is stored, your identity proofs expire after a set period, ensuring privacy even if you stop using the service.",
      },
      {
        question: "Is Tunnel ID open-source?",
        answer:
          "Yes, Tunnel ID is built on an open-source framework, allowing developers to integrate and audit its security model.",
      },
      {
        question: "Where can I get support for Tunnel ID?",
        answer:
          "Visit our official website or join our community on Telegram, Discord, and Twitter for support and updates.",
      },
    ],
  },
];

const FAQs = () => {
  const [activeIndex, setActiveIndex] = useState(null);

  const toggleFAQ = (index) => {
    setActiveIndex(activeIndex === index ? null : index);
  };

  return (
    <div className="faqs-container">
      <h2 className="faqs-title">Tunnel ID - Frequently Asked Questions (FAQ)</h2>
      {faqsData.map((section, sectionIndex) => (
        <div key={sectionIndex} className="faq-section">
          <h3 className="faq-category">{section.category}</h3>
          <div className="faqs-list">
            {section.items.map((faq, index) => {
              const questionIndex = `${sectionIndex}-${index}`;
              return (
                <div
                  key={questionIndex}
                  className={`faq-item ${activeIndex === questionIndex ? "active" : ""}`}
                >
                  <div className="faq-question" onClick={() => toggleFAQ(questionIndex)}>
                    {faq.question}
                    <span className={`faq-icon ${activeIndex === questionIndex ? "rotate" : ""}`}>
                      &#9662;
                    </span>
                  </div>
                  {activeIndex === questionIndex && <div className="faq-answer">{faq.answer}</div>}
                </div>
              );
            })}
          </div>
        </div>
      ))}
      <div className="faq-footer">
        For more details, visit{" "}
        <a href="https://www.tunnelid.me" target="_blank" rel="noopener noreferrer">
          www.tunnelid.me
        </a>{" "}
        or contact us at{" "}
        <a href="mailto:team@tunnelid.me">team@tunnelid.me</a>.
      </div>
    </div>
  );
};

export default FAQs;
