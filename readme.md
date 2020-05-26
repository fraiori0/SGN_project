# SGN - Francesco Iori
This repository contains the files used for the implementation of the navigation method proposed in *Liu J, Pu J, Sun L, He Z. **An Approach to Robust INS/UWB Integrated Positioning for Autonomous Indoor Mobile Robots.** Sensors (Basel). 2019;19(4):950. Published 2019 Feb 23. doi:10.3390/s19040950*

---
## Codice
Il codice è contenuto nella cartella */src* ed è composto da 3 files principali, illustrati di seguito.
Riferirsi ai commenti inseriti nel codice per maggiore dettaglio sulle singole funzioni.

---
### robot_navigation.py
Questo file contiene l'implementazione di alcune funzioni di base (es. generazione della matrice skew-simmetrica per il prodotto vettoriale, a partire da un vettore) e delle classi *Navigator* e *Unicycle*, che compongono il vero nucleo del codice.

#### Navigator_State (class)
Ogni oggetto di questa classe contiene le variabili di posizione, velocità, quaternione di orientazione, bias di accelerazione e bias della velocità angolare.
È presente inoltre un metodo (della classe) per eseguire l'update dello stato con un'integrazione approssimata (Eulero); questa funzione può essere utilizzata, ad esempio, per fare la predizione dello stato nominale.

#### Navigator (class)
Questa classe contiene come variabili interne tutte le matrici e i parametri necessari per il funzionamento del filtro. Sono inoltre contenute molteplici istanze di oggetti della classe *Navigator_state*, ciascuna contente le informazioni relative a uno stato: stato nominale, stato dell'errore, stato nominale con navigazione solo dead-reackoning, etc.
I metodi di questa classe sono tutte le funzioni necessarie per replicare il funzionamento del filtro; per maggiore dettaglio sull'implementazione di ogni metodo (funzione) consultare i commenti contenuti nel codice.
L'ordine in cui chiamare queste funzioni durante la simulazione è visibile in *simulate_and_show.py* e segue lo schema di funzionamento illustrato nel paper (fig. 5, pag.12). 
In un ciclo di funzionamento generico saranno da compiere le seguenti azioni:
```python
am_data,wm_data = uni.navig.generate_INS_measurement(vd_body,w_body)
uni.navig.INS_predict_xn_nominal_state(dt_INS,am_data,wm_data)
uni.navig.INS_predict_x_INS(dt_INS,am_data,wm_data)
if UWBbool:
    # UWB measurement
    UWB_data = uni.navig.generate_UWB_measurement(p,interference=[-0.2,0.2])
    uni.navig.UWB_measurement(dt_UWB,UWB_data.copy())
    uni.navig.compute_z()
    uni.navig.update_Q_TMP(dt_UWB)
    uni.navig.update_F(dt_UWB,uni.navig.am_prev.copy(),uni.navig.wm_prev.copy())
    uni.navig.predict_P_error_state_covariance(dt_UWB)
    # Update innovation
    uni.navig.update_epsilon_innovation()
    uni.navig.update_S_theoretical_innovation_covariance()
    uni.navig.update_Sn_estimated_innovation_covariance()
    # Outlier detection
    uni.navig.update_D_theoretical_zzT_expectation()
    outlier_detected = uni.navig.check_for_outlier()
    if (outlier_detected):
        print("DETECTED")
        uni.navig.update_epsilon_innovation(hold=True)
        uni.navig.update_Sn_estimated_innovation_covariance()
        uni.navig.update_D_theoretical_zzT_expectation()
    # Fuzzy filter
    uni.navig.apply_fuzzy_filter(no_fuzzy=False)
    # Estimate Measurement Noise Covariance
    uni.navig.update_Rn_theoretical_estimated_MNC()
    uni.navig.update_R_estimated_MNC()
    # Compute Kalman Gain and Update error state
    uni.navig.update_Kalman_gain()
    uni.navig.update_dx_and_P()
    # Update nominal state and reset
    uni.navig.update_xn()
    uni.navig.reset_dx_error_state()
```

#### Unycicle_State

Questa classe svolge una funzione simile a *Navigator_State*.

Contiene lo stato di un uniciclo e i metodi per simularlo in tempo discreto con integrazione tramite approssimazione di Eulero. I metodi sono due, uno per una simulazione dinamica, in cui vengono passate le azioni delle forze esterne, e uno cinematico, in cui vengono passati direttamente i valori della velocità e della velocità angolare. Quest'ultima funzione ha la scopo di consentire di concentrarsi solo sulla navigazione e generare traiettorie predeterminate, senza dover svolgere la parte di controllo dell'uniciclo con i metodi implementati da *Unicycle*.

È inoltre presente il metodo *return_as_3D_with_quat()*, che riarrangia lo stato dell'uniciclo in modo che sia compatibile con uno stato navigazione di tipo *Navigator_State*. Questa funzione è necessaria in quanto lo stato dell'uniciclo è implementato in 2D per semplicità di simulazione, mentre *Navigator_State* si riferisce ad una dinamica 3D, in modo da poter essere applicato a qualsiasi caso generale. Durante la simulazione il sistema di navigazione lavora per tracciare una posizione 3D, senza "sapere" di essere applicato a un veicolo che si muove su uno spazio bidimensionale (pavimento).

L'uniciclo è simulato e controllato riferendosi alle equazioni riportate nel corso di Controllo dei Robot del Prof. A.Bicchi, Università di Pisa.

#### Unicycle (class)
Questa classe contiene variabili che definiscono lo stato dell'uniciclo, i suoi parametri (massa, etc.), i parametri del controllore e un oggetto della classe *Navigator*, che rappresenta il sistema di navigazione montato sull'uniciclo.
Contiene i metodi necessari per il controllo in back-stepping dell'uniciclo e il metodo *draw_artists()* per generare i singoli frame di un video, nel caso si voglia salvare un'animazione della navigazione.

---
### simulate_and_show.py
Questo file contiene lo script che svolge la simulazione. È organizzato nel seguente modo:

1. Inizializzazione dei parametri di simulazione (durata totale, frequenza di update di UWB e INS, etc.)
2. Generazione della traiettoria desiderata (utile in caso di controllo in back-stepping dell'uniciclo)
3. Inizializzazione dell'uniciclo e del sistema di navigazione:
    a. Parametri del filtro
    b. Controllo dell'uniciclo
    c. Stato iniziale del sistema di navigazione (qua può essere inserito un errore iniziale nella stima della posizione)
4. Inizializzazione del salvataggio di un'animazione (se *save_video* viene impostato uguale a *True*) e delle variabili in cui eseguire lo storage dei dati di navigazione, da usare per generare i grafici.
5. Simulazione
    $\forall$ step:
    a. Controllo dell'uniciclo, generazione della posizione reale
    b. Controllo del timing dei sotto-sistemi (UWB e INS hanno frequenze diverse di generazione delle misure; cfr. paper)
    c. Sistema di navigazione: l'ordine della chiamata delle funzioni della classe *Navigator* è stato mantenuto equivalente a quello illustrato nel paper. Per maggiore dettaglio vedere grafico a pagina 12 del paper e commenti nel codice.
    d. Salvataggio dei dati generati durante lo step in corso
    e. Eventuale salvataggio del frame video
    f. Avanzamento della simulazione
6. Tracciamento dei grafici. Se la variabile *save_figs* è impostata a *True* i grafici vengono salvati in automatico nella cartella *./graphs*

---
### filter_tuning.py
Questo file svolge un'ottimizzazione Bayesiana per trovare dei valori ottimali dei parametri del filtro. L'implementazione è basata sulla liberia *HyperOpt*.

**simulate()**
Funzione che, dato in ingresso un set di parametri, svolge con esso un numero di simulazioni uguale alla variabile di input *folds*, facendo durare ogni simulazione per un tempo *t_tot*. In output restituisce il valore medio della funzione di perdita e la varianza tra le varie prove. Questi valori vengono utilizzati dalla funzione *fmin* di *HyperOpt*

**Bayesian_tuning()**
Funzione che svolge un'ottimizzazione Bayesiana per un numero di prove uguale a *iterations*.
Se *save* è uguale a *True* il risultato viene salvato in un file *\*.csv* dal nome *filename*.
I parametri *retrieve_trial* e *save_trial* consentono di salvare anche la variabile di tipo *Trials* generata da HyperOpt, in modo da poter continuare successivamente l'ottimizzazione, aumentando il numero di iterazioni. 
**NOTA**: La prima volta che si utilizza un nuovo *filename*, *retrieve_trial* deve essere impostato a *False* (in quanto non esistono ancora prove effettuate già salvate).

La variabile *params* passata in ingresso a Bayesian_tuning() contiene lo spazio dei valori in cui possono essere scelti i parametri.