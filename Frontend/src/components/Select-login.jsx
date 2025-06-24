import React from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Select-login.css"; // Import external CSS file

const credentials = [
  { id: 1, title: "Fingerprint", color: "#4CAF50", icon: "🔒", redirect: "/Verification" },
  { id: 2, title: "Face ID", color: "#03A9F4", icon: "🙂", redirect: "/FacialVerification" },
  { id: 3, title: "GitHub", color: "#333", icon: "🐙", blur: true },
  { id: 4, title: "Email", color: "#007BFF", icon: "📧", blur: true },
];

const LoginCards = () => {
  const navigate = useNavigate();

  const handleCardClick = (cred) => {
    if (cred.redirect) {
      navigate(cred.redirect); // Redirect to the specified path
    }
  };

  return (
    <div className="container-1">
      <h2 className="heading-1">Choose a Login Method</h2>
      <div className="card-grid">
        {credentials.map((cred) => (
          <div
            key={cred.id}
            className={`card ${cred.blur ? "blurred" : ""}`}
            style={{ backgroundColor: cred.color }}
            onClick={() => handleCardClick(cred)}
          >
            <div className="icon">{cred.icon}</div>
            <p className="title">{cred.title}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default LoginCards;
