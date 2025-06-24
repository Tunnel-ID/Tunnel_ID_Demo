import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import HeroSection from "./components/HeroSection";
import Login from "./components/Login";
import IdentityForm from "./components/IdentityForm"; // Import IdentityForm
import Dashboard from "./components/Dashboard";
import TrimelIDPage from "./components/TrimelIDPage";
import UserIdPage from "./components/UserIdPage";
import "./styles/App.css";
import AboutPage from "./components/AboutPage";
import Footer from "./components/Footer";
import Application from "./components/Applications";
import Ecosystem from "./components/Ecosystem";
import TermsAndConditions from "./components/Terms&Conditions";
import FacialCredentialForm from "./components/FacialCredentialForm"; // Import FacialCredentialForm
import LoginCards from "./components/Select-login";
import Verification from "./components/Verification";
import FacialVerification from "./components/FacialVerification";
import FAQs from "./components/FAQS";

function App() {
  return (
    <Router>
      <div className="app-container">
      
        <Routes>
          <Route path="/" element={
            <>
              <HeroSection />
              <Footer></Footer>
             
            </>
          } />
          <Route path="/login" element={<Login />} />
          <Route path="/trimel-id" element={<TrimelIDPage />} />
          <Route path="/identity-form" element={<IdentityForm />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/About" element={<AboutPage />} />
          <Route path="/application" element={<Application />} />
          <Route path="/user/:userId" element={<UserIdPage />} />
          <Route path="/eco-system" element={<Ecosystem />} />
          <Route path="/terms-and-conditions" element={<TermsAndConditions />} />
          <Route path="/facial-credential-form" element={<FacialCredentialForm />} />
          <Route path="/Select-login" element={<LoginCards />} />
          <Route path="/verification" element={<Verification />} />
          <Route path="/FacialVerification" element={<FacialVerification />} />
          <Route path="/faqs" element={<FAQs />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
