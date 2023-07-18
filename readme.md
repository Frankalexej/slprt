# SLPRT: Sign Language Praat Project

## How to run through the codes
1. run paths.py
2. put the videos under src/vid/xxx/
3. run lm_detect.py
4. run graph_extract.py
5. run extract_plot.py

## Directory structure
This project is organized in this way
```
slprt
├── src
│   ├── src_vid
│   ├── dest_vid
│   ├── grph
│   ├── embed
├── codes
```
Only codes will be commited and synced across devices, as these are small files and will be modified heavily. The srcs will are too large to be hosted on github, therefore, they will be transmitted separately, probably through Onedrive or portable USB. 



## src structure
```
.
├── det
│   ├── JapS-AQ4_Resultant
│   │   ├── JapS-AQ4_bon
│   │   ├── JapS-AQ4_booboo
│   │   ├── JapS-AQ4_chokichoki
│   │   ├── JapS-AQ4_gorogoro
│   │   ├── JapS-AQ4_kachikachi
│   │   ├── JapS-AQ4_kirakira
│   │   ├── JapS-AQ4_kyutto
│   │   ├── JapS-AQ4_nebaneba
│   │   ├── JapS-AQ4_panpan
│   │   ├── JapS-AQ4_perapera
│   │   ├── JapS-AQ4_pikapika
│   │   ├── JapS-AQ4_pon
│   │   ├── JapS-AQ4_tonton
│   │   ├── JapS-AQ4_tsuntsun
│   │   ├── JapS-AQ4_yurayura
│   │   └── JapS-AQ4_zuratto
│   └── KorS-AQ4_Resultant
│       ├── KorS-AQ4_cheolcheol
│       ├── KorS-AQ4_chingching
│       ├── KorS-AQ4_eongeong
│       ├── KorS-AQ4_eongeumeongeum
│       ├── KorS-AQ4_geunjeokgeunjeok
│       ├── KorS-AQ4_gobulgobul
│       ├── KorS-AQ4_heumeulheumeul
│       ├── KorS-AQ4_jengjeng
│       ├── KorS-AQ4_mulleongmulleong
│       ├── KorS-AQ4_panjakbanjak
│       ├── KorS-AQ4_peolleokbeolleok
│       ├── KorS-AQ4_podeulbodeul
│       ├── KorS-AQ4_singsing
│       ├── KorS-AQ4_songeulsongeul
│       ├── KorS-AQ4_tengteng
│       └── KorS-AQ4_tungeuldungeul
├── img
├── mediapipe
├── rend
│   ├── pic
│   │   ├── JapS-AQ4_Resultant
│   │   │   ├── JapS-AQ4_bon
│   │   │   ├── JapS-AQ4_booboo
│   │   │   ├── JapS-AQ4_chokichoki
│   │   │   ├── JapS-AQ4_gorogoro
│   │   │   ├── JapS-AQ4_kachikachi
│   │   │   ├── JapS-AQ4_kirakira
│   │   │   ├── JapS-AQ4_kyutto
│   │   │   ├── JapS-AQ4_nebaneba
│   │   │   ├── JapS-AQ4_panpan
│   │   │   ├── JapS-AQ4_perapera
│   │   │   ├── JapS-AQ4_pikapika
│   │   │   ├── JapS-AQ4_pon
│   │   │   ├── JapS-AQ4_tonton
│   │   │   ├── JapS-AQ4_tsuntsun
│   │   │   ├── JapS-AQ4_yurayura
│   │   │   └── JapS-AQ4_zuratto
│   │   └── KorS-AQ4_Resultant
│   │       ├── KorS-AQ4_cheolcheol
│   │       ├── KorS-AQ4_chingching
│   │       ├── KorS-AQ4_eongeong
│   │       ├── KorS-AQ4_eongeumeongeum
│   │       ├── KorS-AQ4_geunjeokgeunjeok
│   │       ├── KorS-AQ4_gobulgobul
│   │       ├── KorS-AQ4_heumeulheumeul
│   │       ├── KorS-AQ4_jengjeng
│   │       ├── KorS-AQ4_mulleongmulleong
│   │       ├── KorS-AQ4_panjakbanjak
│   │       ├── KorS-AQ4_peolleokbeolleok
│   │       ├── KorS-AQ4_podeulbodeul
│   │       ├── KorS-AQ4_singsing
│   │       ├── KorS-AQ4_songeulsongeul
│   │       ├── KorS-AQ4_tengteng
│   │       └── KorS-AQ4_tungeuldungeul
│   └── vid
│       ├── JapS-AQ4_Resultant
│       └── KorS-AQ4_Resultant
└── vid
    └── vid
        ├── JapS-AP1_Resultant
        ├── JapS-AQ4_Resultant
        └── KorS-AQ4_Resultant

```