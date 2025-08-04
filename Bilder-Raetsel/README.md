# Riddle Game mit TensorFlow.js und Vite

Dieses Projekt ist ein React-basierter Rätsel-Detektor, der TensorFlow.js und das COCO-SSD Modell verwendet, um Objekte in der Kamera zu erkennen und mit Rätseln zu verknüpfen.

---

## Voraussetzungen

- Node.js (empfohlen Version 16+)
- npm (wird mit Node.js installiert)

---

## Installation

1. Repository klonen oder Projektordner erstellen
2. Im Projektverzeichnis die Abhängigkeiten installieren:

```bash
# 1. Repository klonen (dies ist die URL für das gesamte Projekt)
git clone https://github.com/Misterz1x/Praktikantentage.git

# 2. In das Projektverzeichnis wechseln
cd dein-repo 

# 3. Abhängigkeiten installieren (inkl. React, TensorFlow.js, coco-ssd und Vite)
npm install
npm install react react-dom
npm install @tensorflow/tfjs @tensorflow-models/coco-ssd
npm install vite

# 4. Entwicklungsversion starten (Vite Dev-Server)
npm run dev

```
Danach öffnet sich das Projekt unter http://localhost:5173.


### Verwendete Technologien
React 18

Vite (Build-Tool & Dev-Server)

TensorFlow.js

COCO-SSD Modell zur Objekterkennung

### Hinweise
Webcam-Zugriff wird benötigt.

Stelle sicher, dass du in einer sicheren Umgebung (https oder localhost) arbeitest, damit der Kamera-Zugriff funktioniert.

