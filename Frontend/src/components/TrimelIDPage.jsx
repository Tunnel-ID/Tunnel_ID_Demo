import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/TrimelIdPage.css";
import backgroundVideo from "../assets/Background.mp4";
import { Buffer } from "buffer";



async function runKeygenFromEmbedding(embeddingList, tag) {

    try{
      console.log("🔑 Running keygen with tag:", tag);

      if (embeddingList.length !== 512) {
        throw new Error(`Embedding must be 512 elements long, got ${embeddingList.length}`);

      }
      const module = await import(`${import.meta.env.BASE_URL}pkg/tunnelid_core.js`);
      await module.default(); // Ensure WASM is initialized
      const { TunnelIDCore } = module;

      const core = new TunnelIDCore(512, 40.2, 10.0, 0.0001);
      const embedding = new Float64Array(embeddingList);
      const result = core.keygen(embedding, tag);

      const keyHex = Buffer.from(result.key).toString("hex");
      console.log("🔑 Key generated successfully:", keyHex);

      const sketchBase64 = Buffer.from(new Float64Array(result.sketch).buffer).toString("base64");
      console.log("🔑 Sketch Base64:", sketchBase64);

      const g_aBase64 = result.g_a; // Assuming g_a is already in Base64 format
      const betaBase64 = result.beta; // Assuming beta is already in Base64 format
      
      console.log("***** g_a raw bytes *****", g_aBase64);
      console.log("****** g_a decoded length ******", Buffer.from(g_aBase64, "base64").length);
      console.log("****** beta decoded length ******", Buffer.from(betaBase64, "base64").length);


      
      const response = await fetch("http://localhost:5001/save_facekey", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          wallet_address: tag,
          sketch_base64: sketchBase64,
          g_a: g_aBase64,
          beta: betaBase64,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Failed to save facekey: ${errorData.message}`);
      }

      alert("✅ Key generated and saved successfully!"); 
      console.log("✅ Key generation and saving completed.");

    } catch (error) {
      console.error("❌ Error in runKeygenFromEmbedding:", error);
      alert("❌ Key generation failed: " + error.message);
    }
}


const TrimelIDPage = () => {
  const navigate = useNavigate();
  const [walletAddress, setWalletAddress] = useState("No wallet connected");
  const [sketchHex, setSketchHex] = useState(null);
  const [sketchBase64, setSketchBase64] = useState(null);
  const [isFingerprintRegistered, setIsFingerprintRegistered] = useState(false);
  const [expirationDate, setExpirationDate] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [stream, setStream] = useState(null);
  const [capturedImages, setCapturedImages] = useState([]);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // ✅ Initialize wallet and localStorage data
  useEffect(() => {
    const initialize = async () => {
      let address = localStorage.getItem("walletAddress");

      if (!address && window.ethereum) {
        const accounts = await window.ethereum.request({ method: "eth_requestAccounts" });
        if (accounts.length > 0) {
          address = accounts[0];
          localStorage.setItem("walletAddress", address);
        }
      }

      if (address) {
        setWalletAddress(address);
        const saved = localStorage.getItem(`credentials_${address}`);
        if (saved) {
          try {
            const parsed = JSON.parse(saved);
            setSketchHex(parsed.sketchHex || null);
            setSketchBase64(parsed.sketchBase64 || null);
            setExpirationDate(parsed.expirationDate || null);
            setIsFingerprintRegistered(parsed.isFingerprintRegistered || false);
          } catch (e) {
            console.error("Credential parsing failed:", e);
          }
        }
      }
    };

    initialize();
  }, []);

  const generateExpirationDate = () => {
    const date = new Date();
    date.setFullYear(date.getFullYear() + 3);
    return date.toLocaleDateString("en-US", { month: "2-digit", year: "2-digit" });
  };

  const handleCreateFacialCredential = () => {
    setCapturedImages([]);
    setIsCameraActive(true);
    startCamera();
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      alert("Camera access failed. Please allow permissions.");
    }
  };

  const stopCamera = () => {
    if (stream) stream.getTracks().forEach(track => track.stop());
    setStream(null);
    setIsCameraActive(false);
    setCapturedImages([]);
  };

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas) {
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = canvas.toDataURL("image/png");

      setCapturedImages(prev => {
        const updated = [...prev, imageData];
        if (updated.length === 2) {
          sendImageToBackend(updated[0], updated[1]);
        } else {
          alert("Face 1 captured. Please capture Face 2.");
        }
        return updated;
      });
    }
  };

  const sendImageToBackend = async (image1, image2) => {
    try {
      const response = await fetch("http://localhost:5001/register_facial", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image1, image2 }),
      });

      const data = await response.json();
      if (!response.ok) {
        alert(`Facial registration failed: ${data.message}`);
        return;
      }

      const expiration = generateExpirationDate();
      setSketchBase64(data.sketch_base64);
      setSketchHex(data.sketch_hex);
      setExpirationDate(expiration);

      localStorage.setItem(`credentials_${walletAddress}`, JSON.stringify({
        sketchHex: data.sketch_hex,
        sketchBase64: data.sketch_base64,
        expirationDate: expiration,
        isFingerprintRegistered
      }));

      alert("Facial credential registered successfully.");
      stopCamera();
    } catch (error) {
      alert("Facial registration failed. Please try again.");
    }
  };

  const handleRegisterFingerprint = async (event) => {
    const file = event.target.files[0];
    if (!file) return alert("Please upload a fingerprint file.");

    const formData = new FormData();
    formData.append("fingerprint_file", file);
    formData.append("wallet_address", walletAddress);

    try {
      const response = await fetch("http://localhost:5001/register_fingerprint", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        alert(`Fingerprint registration failed: ${data.message}`);
        return;
      }

      const expiration = generateExpirationDate();
      setSketchHex(data.user_id);
      setExpirationDate(expiration);
      setIsFingerprintRegistered(true);

      localStorage.setItem(`credentials_${walletAddress}`, JSON.stringify({
        sketchHex: data.user_id,
        sketchBase64,
        expirationDate: expiration,
        isFingerprintRegistered: true
      }));

      alert("Fingerprint credential registered successfully.");
    } catch (error) {
      alert("Fingerprint registration failed. Please try again.");
    }
  };

  const handleRefresh = () => {
   window.location.reload();  
  };

  const sendToProcessFace = async () => {
    console.log("🧠 Button Clicked!");
  
    const video = videoRef.current;
    const canvas = canvasRef.current;
  
    if (!video || !canvas) {
      alert("Camera is not active. Please start the camera first.");
      return;
    }
    
    if (video.readyState < 2) {
      alert("Video is not ready. Please wait a moment."); 
      return;
    }
  
    // Ensure canvas matches video resolution
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    console.log("Video size:", video.videoWidth, video.videoHeight);
  
    const ctx = canvas.getContext("2d");
    // ✅ Reset transform before draw
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  
    // ✅ Mirror if needed (comment out if not mirrored UI)
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  
    // ✅ Draw the actual video frame
    await new Promise(requestAnimationFrame);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const isWhite = pixels.every((v, i) => (i + 1) % 4 === 0 || v > 250);
    if (isWhite) {
      alert("Frame is blank or too light. Please ensure the camera is capturing properly.");
      return;
    }
  
    // ✅ Extract Base64 data
    const imageData = canvas.toDataURL("image/png", 0.95);
    console.log("🖼️ Image data captured:", imageData);  // Debug: Log first 100 chars
    
    
    if (imageData.length < 500){
      alert("❌ Image data is too small. Please ensure the camera is capturing properly.");
      return;
    }
    console.log("🖼️ Frontend Image data length:", imageData.length);
    // Debug: view image
    //window.open(imageData);  // <-- Optional
  
    console.log("🧠 Sending image to backend...");

    try {
      const response = await fetch("http://localhost:5001/process_face", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });

      const data = await response.json();
      if ( !response.ok || data.status !== "success"){
        throw new Error(data.message || "Face embedding failed");
      }

      let embedding = data.embedding;

      if (Array.isArray(embedding[0])){
        embedding = embedding.flat(); // Flatten if nested
      }
      console.log("🧠 Embedding Flatened:", embedding);
      if (embedding.length !== 512) {
        alert(`❌ Embedding must be 512 elements long, got ${embedding.length}`);
        throw new Error(` ❌ Embedding must be 512 elements long, got ${embedding.length}`);
      }

      console.log(" ✅ Face Embedding received:", embedding);
      await runKeygenFromEmbedding(embedding, walletAddress);
    } catch(error) {
      console.error("❌ Error processing face:", error);
      alert("❌ Face processing failed: " + error.message);
    }
  };
async function recoverFromServerData(embeddings, sketch_base64, g_a_b64, beta_b64, tag ){
    const module = await import(`${import.meta.env.BASE_URL}pkg/tunnelid_core.js`);
    await module.default(); // Ensure WASM is initialized
    const { TunnelIDCore } = module;
    const core = new TunnelIDCore(512, 40.2, 10.0, 0.0001);

    const sketchArray = new Float64Array(Buffer.from(sketch_base64, 'base64').buffer);
    const sketchJsonObj = {
      key: Array(32).fill(0),
      sketch: Array.from(sketchArray),
      g_a: g_a_b64,
      beta: beta_b64,
      tag,
     };
      
     const sketch_json = JSON.stringify(sketchJsonObj);
     console.log("🗝️ Sketch JSON:", sketch_json);
     console.log("🗝️ Sketch JSON length:", sketch_json.length);
     console.log("Embedding length", embeddings.length);
     const recoveredKey = core.keyrecover(embeddings, sketch_json);
     if(!recoveredKey){
        throw new Error("Key recovery failed. Please ensure the embeddings and sketch are correct.");
     }

     return Buffer.from(recoveredKey).toString("hex");
                                
}
  const handleKeyRecovery = async () => {
    if (!videoRef.current || !canvasRef.current) {
      alert("Camera is not active. Please start the camera first.");
      return;
    }
    if (videoRef.current.readyState < 2) {
      alert("Video is not ready. Please wait a moment.");
      return;
    }
  
    canvasRef.current.width = videoRef.current.videoWidth;
    canvasRef.current.height = videoRef.current.videoHeight;
    const ctx = canvasRef.current.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.translate(canvasRef.current.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);

    const imageData = canvasRef.current.toDataURL("image/png", 0.95);
    console.log("🖼️ Image data captured for recovery:", imageData);

    const walletAddress = localStorage.getItem("walletAddress");
    if (!walletAddress) { 
      alert("No wallet address found. Please connect your wallet first.");
      return;
    }

    try{
      const embedRes = await fetch("http://localhost:5001/process_face", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });

      const embedData = await embedRes.json();
      if (embedData.status !== "success"){
        throw new Error(embedData.message || "Face embedding failed");
      }

      const embedding = embedData.embedding.flat();
      if (embedding.length !== 512) {
        throw new Error(`Embedding must be 512 elements long, got ${embedding.length}`);
      }

      const recRes = await fetch(`http://localhost:5001/recover_facekey`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tag: walletAddress }),
      });
      const recData = await recRes.json();
      if (recData.status !== "success") {
        throw new Error(recData.message || "Face key recovery failed");
      }

      const recoveredKey = await recoverFromServerData(embedding, recData.sketch_base64, recData.g_a, recData.beta, walletAddress);
      alert(`You are a verified Human!`);
    }
    catch (error) {
      console.error("❌ Error during key recovery:", error);
      alert("❌ Key recovery failed: " + error.message);
    }
  }

      

  return (
    <div className="trimel-id-page">
      <video autoPlay loop muted className="background-video">
        <source src={backgroundVideo} type="video/mp4" />
      </video>

      <div className="content-overlay">
        <button className="refresh-button" onClick={handleRefresh}>
          <span className="refresh-icon">🔄</span> Refresh
        </button>

        <h2 className="wallet-info">Metamask Public Key: {walletAddress}</h2>

        <div className="credential-grid">
          <div className="credential-card">
            <h3>Create Fingerprint Credential</h3>
            <input type="file" onChange={handleRegisterFingerprint} />
            {isFingerprintRegistered && <p className="badge">✅ Registered</p>}
          </div>

          <div className="credential-card" onClick={handleCreateFacialCredential}>
            <h3>Create Facial Credential</h3>
            {sketchBase64 && <p className="badge">✅ Registered</p>}
          </div>
          <div className="credential-card coming-soon"> Voice Credential </div>
          <div className="credential-card coming-soon"> Iris Credential </div>
        </div>



        {sketchBase64 && (
          <div className="credential-status">
            <h4>Facial ID</h4>
            <p>{sketchBase64}</p>
            <p>Expires: {expirationDate}</p>
          </div>
        )}

        {isCameraActive && (
          <div className="camera-popup">
            <div className="camera-content">
              <video ref={videoRef} width="640" height="480" autoPlay></video>
              <div className="camera-buttons">
                <button onClick={sendToProcessFace} className="process-face-button">Generate Key</button>
                <button onClick={stopCamera} className="close-camera-button">Close</button>
                <button onClick={handleKeyRecovery} className="process-face-button">Proof you are Human!</button>
              </div>
            </div>
          </div>
        )}

        <canvas ref={canvasRef} width="640" height="480" style={{ display: "none" }}></canvas>
      </div>
    </div>
  );
};

export default TrimelIDPage;
