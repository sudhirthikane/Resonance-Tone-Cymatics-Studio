A desktop Python application for generating mantra-based audio sessions, mixing them with mono or binaural tones, previewing waveform and cymatics-style visuals, and exporting WAV/PNG assets for meditation, sound design, and spiritual content workflows.

Features
Preset-based mantra session generation, including built-in Shreem presets.

Creative text-to-frequency mapping for quick experimental tone generation.

WAV mantra upload support.

Microphone recording support for creating custom mantra sessions.

Mono and binaural background tone generation.

Tone-only preview and mantra + tone mix preview.

Full-session WAV export.

Dedicated 5-minute preset WAV export.

Cymatics-style pattern visualization and PNG export.

Scrollable desktop UI with visible disabled actions during background processing.

Progress bar with percentage updates during compute, preview, recording, and export tasks.

Screens and workflow
The app is designed as a single-window desktop studio:

Choose a preset or enter a custom mantra keyword.

Adjust the base frequency and repetition count.

Upload a WAV mantra or record from the microphone.

Select background mode: off, mono, or binaural.

Preview the tone or the full mantra + tone mix.

Export a full session WAV, a fixed 5-minute WAV, or a cymatics PNG.

Tech stack
Python

Tkinter

NumPy

SciPy

Matplotlib

sounddevice (optional, for recording)

Installation
1. Clone the repository
bash
git clone https://github.com/your-username/resonance-tone-cymatics-studio.git
cd resonance-tone-cymatics-studio
2. Create a virtual environment
bash
python -m venv venv
3. Activate the environment
Windows
bash
venv\Scripts\activate
macOS / Linux
bash
source venv/bin/activate
4. Install dependencies
bash
pip install numpy matplotlib scipy sounddevice
Run the app
bash
python improved_cymatics_dashboard_v3.py
Build Windows EXE
If you want a standalone Windows executable:

bash
pyinstaller --noconfirm --onefile --windowed --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide2 --exclude-module PySide6 improved_cymatics_dashboard_v3.py
The generated executable will be available in the dist/ folder.

Project structure
text
.
├── improved_cymatics_dashboard_v3.py
├── README.md
└── dist/                # generated after packaging
Use cases
Meditation audio creation

Mantra chanting session generation

Spiritual / wellness content production

Focus and relaxation sound experiments

Binaural beat prototyping

Visual content generation for thumbnails or branding

Notes
Binaural mode is intended for stereo headphone listening.

The text-to-frequency conversion is a creative mapping, not a scientific or traditional fixed-frequency system.

Microphone recording requires sounddevice and a working PortAudio setup.

Future improvements
MP3 export

Session save/load presets

More mantra packs and themed frequency banks

Better installer and auto-update support

Multi-language interface

macOS and Linux packaging guides

License
Choose a license before publishing publicly. MIT is a simple default if you want broad reuse.

