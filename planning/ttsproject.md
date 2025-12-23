# Project

## Task

actually coordinate everything

- book
- tts
  - tts
  - lexicon
- audio conversion
- written files

## What it does

- book
  - creates book object
  - reads chapters from book object
- tts
  - either
    - accepts external pipeline
    - creates new pipeline
  - makes lexicon from paths
  - runs tts with specified parameters

## Step by step

### Init

Init should do everything needed for the project to be reopened without any arguments

- Move epub into project folder
- Add project.yaml to project folder
  - Title
  - Author
  - Series
  - Progress flags (all False)
- Make Book from epub path
- Write full text to book.txt
- Make Lexicon from lexicon yaml paths
- write lexicon to lexicon.yaml (Should it be this or should it keep a list of references?)
- Write to project.yaml
  - Epub path
  - (?) Lexicon paths if we're not making a project-specific one
  - Voice
  - Speed
  - Init flag True

### Splits

- Change g2p to Lexicon's g2g2p
- for each chapter in Book
  - tts entire chapter
  - calculate length
  - save chapter to split file
  - write split (chapter title,length,filename) to project.yaml>audio
- Restore original g2p
- split flag True

### Master

- generate ffmpeg concat file
- merge all splits into master using ffmpeg
  - (?) normalize volume
  - (?) compress levels
  - compress data
- write master filename to project.yaml
- master flag true

### M4B

- generate ffmpeg splits data file
- convert master to m4b
- write m4b filename to project.yaml
- m4b flag true

### Copy

- try copy to destination
- (?) write copy locations to project.yaml
- copy flag true

## Inputs

- book
  - path to epub
  - forced metadata
    - Title
    - Series
    - Author
- tts
  - TTS Obj with pipeline (optional)
  - Lexicon
    - g2g sources
    - g2p sources
  - Voice
  - Speed
  - Output location

## project.yaml

Title
(Author)
(Series)

- Progress

  - (? If done well, this will just be a helper or a way to skip steps)
  - init
  - splits
  - master
  - m4b
  - copy

- Book

  - EPUB Path

- Lexicon

  - G2G Sources
  - G2P Sources

- TTS

  - Voice
  - Speed

- Audio

  - Audio split obj
    - Filename
    - Title
    - Duration
  - Audio master file
    - Filename
    - ?
  - FFMPEG metadata file (? This might be best to just generate every time)
  - M4B

- Files

  - Output location
