AudiobookProject

Config File

- name
- author (override)
- series (override)
- file globs
- output dir
- voice
- speed
- g2g
- g2p

Get list of paths

ProjectConfig

- init_path
- author (override)
- series (override)
- output dir
- voice
- speed
- g2g paths
- g2p paths

AudiobookProject.open(dir,ProjectConfig)

- config

???

- Fully captures the configuration and the current state of the AudiobookProject
- Config
  - from arguments
  - from project.yaml
    - maybe supplemented by arguments
    - maybe superceded by arguments
- State

dir structure

- Project/
  - project.json
  - state.json
  - ?records
    - chapters.json
  - artifacts/
    - chapter_wavs/
      - chapter_n.wav
    - master.m4a
    - ## final
