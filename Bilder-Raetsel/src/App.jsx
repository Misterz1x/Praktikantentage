import React, { useEffect, useRef, useState, useMemo } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";
import "./App.css";

const riddles = [
  { text: "Ich trage dich auf zwei Rädern, aber ich bin nicht motorisiert.", answer: "bicycle" },
  { text: "Ich öffne mich, damit du an meine kalten Vorräte kommst.", answer: "refrigerator" },
  { text: "Ich schneide Papier, doch ich bin kein Messer", answer: "scissors" },
  { text: "Ich bewege mich durch die Luft, oft höher als Wolken und weiter als Vögel.", answer: "airplane" },
  { text: "Ich begleite dich auf Wanderungen, gefüllt mit dem, was du brauchst.", answer: "backpack" },
  { text: "Ich halte Flüssigkeiten, manchmal mit einem Henkel, manchmal ohne.", answer: "cup" },
  { text: "Ich bin krumm, gelb und Affen lieben mich.", answer: "banana" },
  { text: "Ich habe viele Tasten und du benutzt mich zum Schreiben.", answer: "keyboard" },
  { text: "Ich bin ein großes Haustier, bellend und treu.", answer: "dog" },
  { text: "Ich kann fliegen und habe bunte Federn.", answer: "bird" }
];

const celebrationGifUrl = "https://media.giphy.com/media/111ebonMs90YLu/giphy.gif"; 

// kleine Shuffle-Funktion
function shuffleArray(array) {
  let arr = [...array];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function App() {
  const videoRef = useRef(null);
  const [current, setCurrent] = useState(0);
  const [status, setStatus] = useState("Show the correct object!");
  const [quizFinished, setQuizFinished] = useState(false);

  // riddles werden EINMAL gemischt, wenn Komponente geladen wird
  const shuffledRiddles = useMemo(() => shuffleArray(riddles), []);

  useEffect(() => {
    if (quizFinished) return; 

    const initCamera = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      const model = await cocoSsd.load();
      detectFrame(videoRef.current, model);
    };

    const detectFrame = (video, model) => {
      model.detect(video).then(predictions => {
        const detectedClasses = predictions.map(p => p.class);
        const expected = shuffledRiddles[current].answer;

        if (detectedClasses.includes(expected)) {
          setStatus(`✅ Correct! Detected: ${expected}`);
          setTimeout(() => {
            if (current + 1 < shuffledRiddles.length) {
              setCurrent(current + 1);
              setStatus("Show the correct object!");
            } else {
              setQuizFinished(true);
              setStatus("🎉 Quiz finished! Well done!");
            }
          }, 2000);
        } else {
          requestAnimationFrame(() => detectFrame(video, model));
        }
      });
    };

    initCamera();
  }, [current, quizFinished, shuffledRiddles]);

  return (
    <div className="app">
      <h1>Riddle Game</h1>
      {!quizFinished ? (
        <>
          <h2>Riddle {current + 1} of {shuffledRiddles.length}</h2>
          <p>{shuffledRiddles[current].text}</p>
          <p>Status: {status}</p>
          <video ref={videoRef} width="400" height="300" autoPlay muted></video>
        </>
      ) : (
        <>
          <p>{status}</p>
          <img src={celebrationGifUrl} alt="Celebration" width="400" />
        </>
      )}
    </div>
  );
}

export default App;
