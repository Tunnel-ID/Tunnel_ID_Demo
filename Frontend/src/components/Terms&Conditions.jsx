import React from "react";
import "../styles/Terms&Conditions.css";

const TermsAndConditions = () => {
  return (
    <div className="terms-page">
      <div className="terms-container">
        <h1 className="terms-title">Terms and Conditions</h1>

        <section className="terms-section">
          <h2 className="section-title">1. Introduction & Acceptance of Terms</h2>
          <p>
            By using our Services, you confirm that you have read, understood, and agreed to these Terms, including any future modifications. These Terms constitute a legally binding agreement between you and Tunnel ID Network.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">2. Definitions</h2>
          <ul>
            <li><strong>Tunnel ID Network:</strong> The decentralized identity platform encompassing Tunnel ID Protocol, Tunnel Wallet, and Tunnel Network.</li>
            <li><strong>Services:</strong> All products, tools, applications, and platforms provided by Tunnel ID Network.</li>
            <li><strong>User/You:</strong> Any person who accesses or uses our Services.</li>
            <li><strong>Account:</strong> A registered profile or identity created by a User to access our Services.</li>
            <li><strong>Biometric Data:</strong> Unique biological characteristics (e.g., facial recognition, fingerprints, iris scans) processed to generate secure cryptographic keys. Note that raw biometric data is never stored.</li>
            <li><strong>Public/Private Key Pair:</strong> Cryptographic keys generated from your biometric inputs, where the private key remains exclusively under your control.</li>
          </ul>
        </section>

        <section className="terms-section">
          <h2 className="section-title">3. Eligibility</h2>
          <p>
            You must be at least 18 years old or of legal age in your jurisdiction to use our Services. By using the Services, you represent and warrant that you meet this requirement.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">4. Account Registration & Security</h2>
          <p>
            <strong>Registration:</strong> To access certain features of our Services, you may be required to create an account. You agree to provide accurate, current, and complete information during registration.
          </p>
          <p>
            <strong>Security:</strong> You are responsible for maintaining the confidentiality of your account credentials and for all activities that occur under your account. Tunnel ID Network employs biometric cryptography to generate secure keys, but you must safeguard your private keys as they are never stored on our servers.
          </p>
          <p>
            <strong>Key Refresh:</strong> Our Tunnel Wallet includes a key refresh feature, which periodically updates your public/private key pairs to enhance security.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">5. Use of Biometric Data & Identity Verification</h2>
          <p>
            <strong>Biometric Processing:</strong> By using our Services, you consent to the processing of your biometric data solely for generating cryptographic keys. Tunnel ID Network uses advanced techniques (e.g., fuzzy signatures, zero-knowledge proofs) to create unforgeable credentials without storing any raw biometric information.
          </p>
          <p>
            <strong>Privacy & Control:</strong> Your biometric data is processed locally and transformed into a secure, decentralized identity. You maintain exclusive control over your private keys and digital identity.
          </p>
          <p>
            <strong>Verification:</strong> Tunnel ID Network verifies your uniqueness and authenticity to ensure one human equals one verified identity, preventing duplicate registrations and fraudulent activities.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">6. Permissible Use & User Responsibilities</h2>
          <p>
            <strong>Lawful Use:</strong> You agree to use our Services only for lawful purposes and in compliance with all applicable laws, regulations, and these Terms.
          </p>
          <p>
            <strong>Prohibited Conduct:</strong> You shall not misuse our Services by engaging in any activity that could harm the platform, compromise its security, or interfere with other users’ experiences. This includes, but is not limited to, attempting to reverse-engineer our biometric cryptography, distributing malware, or engaging in fraudulent transactions.
          </p>
          <p>
            <strong>Data Accuracy:</strong> You are responsible for ensuring that any information you provide is accurate and up to date.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">7. Intellectual Property</h2>
          <p>
            <strong>Ownership:</strong> All content, software, designs, and intellectual property rights in our Services are owned by Tunnel ID Network or its licensors. This includes the underlying technology, biometric algorithms, and user interface elements.
          </p>
          <p>
            <strong>Restrictions:</strong> You may not reproduce, modify, distribute, or create derivative works from any part of our Services without prior written consent from Tunnel ID Network.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">8. Third-Party Services and Links</h2>
          <p>
            Our Services may contain links to third-party websites, applications, or services. Tunnel ID Network is not responsible for the content, privacy policies, or practices of any third parties. Your use of third-party services is governed solely by their respective terms and conditions.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">9. Transactions, Payments, and Digital Assets</h2>
          <p>
            <strong>Unified Wallet Management:</strong> Tunnel Wallet serves as your gateway to the Tunnel Network, enabling you to manage multiple wallets and track all platforms where you have used your Tunnel ID for verification.
          </p>
          <p>
            <strong>Decentralized Transactions:</strong> Transactions executed via our Services occur on decentralized networks. While we implement robust security measures, you acknowledge that risks inherent to blockchain technology exist.
          </p>
          <p>
            <strong>No Storage of Sensitive Keys:</strong> Your private keys remain exclusively in your control, ensuring that your digital transactions and identity verifications are secure and private.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">10. Disclaimer of Warranties</h2>
          <p>
            <strong>“As Is” Basis:</strong> Our Services are provided on an “as is” and “as available” basis. Tunnel ID Network makes no warranties, express or implied, regarding the Services’ performance, accuracy, or reliability.
          </p>
          <p>
            <strong>No Guarantee of Uninterrupted Service:</strong> We do not guarantee that our Services will be error-free, secure, or uninterrupted at all times.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">11. Limitation of Liability</h2>
          <p>
            In no event shall Tunnel ID Network, its affiliates, officers, or employees be liable for any indirect, incidental, special, consequential, or punitive damages arising from your use of the Services. Our total liability, whether in contract, tort, or otherwise, shall not exceed the amount you have paid (if any) for accessing our Services.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">12. Indemnification</h2>
          <p>
            You agree to indemnify, defend, and hold harmless Tunnel ID Network, its affiliates, and their respective officers, directors, employees, and agents from any claims, damages, liabilities, or expenses arising from your use of our Services or violation of these Terms.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">13. Modifications to the Terms</h2>
          <p>
            Tunnel ID Network reserves the right to modify or update these Terms at any time without prior notice. Any changes will be effective immediately upon posting on our website. Your continued use of the Services after any modifications constitutes your acceptance of the updated Terms.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">14. Termination and Suspension</h2>
          <p>
            We reserve the right to suspend or terminate your access to our Services at our sole discretion, without notice, for conduct that we believe violates these Terms or is harmful to other users or the platform. Upon termination, all rights granted to you under these Terms will cease, and you must immediately stop using the Services.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">15. Governing Law and Dispute Resolution</h2>
          <p>
            <strong>Governing Law:</strong> These Terms are governed by and construed in accordance with the laws of the jurisdiction in which Tunnel ID Network operates.
          </p>
          <p>
            <strong>Dispute Resolution:</strong> Any disputes arising out of or related to these Terms shall be resolved through binding arbitration or in the courts of the appropriate jurisdiction, as agreed upon by both parties.
          </p>
          <p>
            <strong>Jurisdiction:</strong> By using our Services, you consent to the exclusive jurisdiction and venue of the courts in the applicable jurisdiction for any disputes arising under these Terms.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">16. Force Majeure</h2>
          <p>
            Tunnel ID Network shall not be liable for any delays or failures in performance due to circumstances beyond our reasonable control, including but not limited to natural disasters, acts of war, or network failures.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">17. Entire Agreement</h2>
          <p>
            These Terms, along with our Privacy Policy and any other legal notices published on our Services, constitute the entire agreement between you and Tunnel ID Network regarding your use of the Services. They supersede all prior and contemporaneous agreements, proposals, or representations, whether written or oral.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">18. Severability</h2>
          <p>
            If any provision of these Terms is found to be invalid or unenforceable by a court of competent jurisdiction, such provision shall be severed, and the remaining provisions will continue in full force and effect.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">19. Waiver</h2>
          <p>
            Failure to enforce any provision of these Terms shall not be deemed a waiver of future enforcement of that or any other provision.
          </p>
        </section>

        <section className="terms-section">
          <h2 className="section-title">20. Contact Information</h2>
          <p>
            For any questions, concerns, or feedback regarding these Terms, please contact us at: <br />
            <strong>Email:</strong> hello@tunnel.me
          </p>
        </section>
      </div>
    </div>
  );
};

export default TermsAndConditions;
