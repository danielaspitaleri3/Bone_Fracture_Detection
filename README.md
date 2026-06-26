# Rilevamento automatico di fratture in radiografie pediatriche

Progetto di **Image Processing and Analysis** e **Machine Learning** per il rilevamento automatico di fratture in radiografie pediatriche dell’avambraccio.

Il sistema lavora a livello di **detection**: analizza la radiografia, genera regioni sospette tramite bounding box, estrae feature numeriche e filtra i falsi positivi con un modello supervisionato.

> Il progetto è pensato come supporto alla revisione e non come sostituto della valutazione medica.

---

## Obiettivo del progetto

L’obiettivo è realizzare una pipeline automatica capace di:

* individuare possibili fratture in radiografie pediatriche;
* generare ROI candidate tramite tecniche di Image Processing;
* descrivere ogni ROI con feature numeriche;
* filtrare i falsi positivi tramite Machine Learning;
* applicare post-processing con Non-Maximum Suppression;
* visualizzare i risultati finali tramite interfaccia grafica.

---

## Dataset

Il progetto utilizza un sottoinsieme del dataset pubblico **GRAZPEDWRI-DX**, composto da radiografie traumatiche pediatriche dell’avambraccio.

Le annotazioni sono fornite in formato **YOLO**.
Nel progetto viene considerata esclusivamente la classe:

```text
class_id = 3  -> fracture
```

Tutte le altre classi vengono ignorate durante training, visualizzazione e valutazione.

Una ROI viene considerata positiva se soddisfa il criterio:

```text
IoU >= 0.50
```

---

## Pipeline generale

```text
Radiografia
   |
   v
Image Processing in C++
   |
   v
ROI candidate + CSV feature
   |
   v
Filtro Machine Learning
   |
   v
Non-Maximum Suppression
   |
   v
Detection finali + GUI
```

La pipeline è divisa in quattro blocchi principali:

1. **Generazione delle ROI candidate**
2. **Estrazione delle feature**
3. **Classificazione tramite Machine Learning**
4. **Post-processing e visualizzazione**

---

## Image Processing

La fase di Image Processing genera automaticamente le regioni sospette.

Le principali operazioni svolte sono:

* normalizzazione della radiografia;
* segmentazione della regione ossea;
* enhancement locale tramite CLAHE e sharpening;
* ricerca di ROI interne ed esterne;
* confronto con le ground truth tramite IoU;
* salvataggio delle ROI e delle feature in file CSV.

Le ROI vengono prodotte da due rami complementari:

| Ramo         | Descrizione                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------- |
| **External** | cerca irregolarità lungo il profilo corticale dell’osso                                      |
| **Internal** | cerca variazioni locali di intensità, contrasto e gradiente all’interno della maschera ossea |

---

## Feature estratte

Per ogni ROI candidata vengono estratte feature numeriche appartenenti a tre famiglie principali:

| Feature  | Descrizione                                                                  |
| -------- | ---------------------------------------------------------------------------- |
| **GLCM** | descrive la texture tramite relazioni di co-occorrenza tra livelli di grigio |
| **LBP**  | cattura micro-pattern locali di intensità                                    |
| **HOG**  | descrive gradienti e discontinuità orientate                                 |

La configurazione finale utilizza la combinazione:

```text
GLCM + LBP + HOG
```

Questa scelta permette di rappresentare meglio le possibili fratture, che possono apparire come variazioni di texture, micro-pattern o discontinuità nei gradienti.

---

## Machine Learning

La fase di Machine Learning ha il compito di ridurre i falsi positivi generati dall’Image Processing.

Sono stati confrontati diversi modelli:

* SVM lineare;
* SVM con kernel RBF;
* KNN;
* Random Forest;
* Ensemble Soft Voting;
* Ensemble Stacking.

Lo split del dataset viene effettuato **a livello di immagine**, evitando che ROI della stessa radiografia finiscano contemporaneamente in training, validation e test.

L’undersampling viene applicato solo al training set, mentre validation e test mantengono la distribuzione reale delle ROI.

Il modello finale selezionato è:

```text
GLCM + LBP + HOG + SVM-RBF
```

---

## Post-processing

Dopo il filtro ML viene applicata la **Non-Maximum Suppression**, utile per eliminare bounding box ridondanti sulla stessa zona anatomica.

Configurazione finale:

```text
Soglia ML: 0.50
Soglia NMS: 0.40
Criterio TP: IoU >= 0.50
```

Se più detection coprono la stessa frattura, viene mantenuta come vero positivo quella con score ML maggiore.

---

## Risultati principali

Il modello finale ottiene sul test set i seguenti risultati:

| Metrica           | Valore |
| ----------------- | -----: |
| Precision         | 0.5468 |
| Recall            | 0.7603 |
| F1-score          | 0.6361 |
| Average Precision | 0.6776 |

Il sistema privilegia il **recall**, scelta coerente con un contesto di supporto alla rilevazione di fratture, dove è importante ridurre il numero di falsi negativi.

---

## Interfaccia grafica

Il progetto include una GUI per la revisione qualitativa dei risultati.

La GUI permette di visualizzare:

* ROI candidate prodotte dall’Image Processing;
* detection finali dopo filtro ML e NMS;
* ground truth YOLO della classe fracture;
* veri positivi, falsi positivi e falsi negativi.

Legenda degli esiti:

| Esito      | Significato                                                |
| ---------- | ---------------------------------------------------------- |
| **TP**     | detection corretta con IoU ≥ 0.50                          |
| **FP**     | detection non corrispondente a una frattura annotata       |
| **FN**     | frattura reale non rilevata                                |
| **IGNORE** | detection duplicata sulla stessa ground truth già rilevata |

---

## Moduli principali

| File                            | Descrizione                                        |
| ------------------------------- | -------------------------------------------------- |
| `main.cpp`                      | menu principale del programma C++                  |
| `functions.cpp` / `functions.h` | implementazione della pipeline di Image Processing |
| `fracture_fp_filter.py`         | training e selezione del modello ML                |
| `evaluate_test_postprocess.py`  | valutazione finale sul test set                    |
| `pipeline_gui.py`               | interfaccia grafica per la revisione dei risultati |
| `fracture_fp_model.pkl`         | modello ML salvato                                 |
| `best_postprocess_params.json`  | parametri finali di post-processing                |


---

## Conclusione

Il progetto mostra come tecniche classiche di **Image Processing** possano essere combinate con modelli di **Machine Learning** per affrontare un problema di detection medica.

La fase IPA genera molte ROI per massimizzare la sensibilità, mentre il modello ML e la NMS riducono falsi positivi e duplicati, producendo un risultato finale più leggibile e valutabile.
