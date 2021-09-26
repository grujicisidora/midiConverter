import numpy as np
import librosa
import midiutil
import sys

#matrica tranzicije - T[i, j] - verovatnoca da ce se iz stanja i preci u stanje j
def transition_matrix(note_min, note_max, p_stay_note, p_stay_silence):
    #note_min - najniza nota obuhvacena programom, u formatu stringa
    #note_max - najvisa nota obuhvacena programom, u formatu stringa
    #p_stay note - verovatnoca da ce se iz stanjaa sustain-a ponovo preci u isto to stanje
    #p_stay_silence - verovatnoca da ce se iz stanja tisine ponovo preci u to stanje 
    midi_min = librosa.note_to_midi(note_min) #konvertuje string notaciju u odgovarajuci MIDI "broj" note - u ovom slucaju najnizeg tona 
    midi_max = librosa.note_to_midi(note_max) #isto kao prethodna linija, samo se odnosi na najvisi ton
    n_notes = midi_max - midi_min + 1 #ukupan broj tonova obuhvacenih programom
    #treba napomenuti da je svakom stanju data jednaka verovatnoca za tranziciju
    p_ = (1-p_stay_silence)/n_notes #verovatnoca da ce se preci iz stanja tisine u stanje onseta za svaku notu 
    p__ = (1-p_stay_note)/(n_notes+1) #verovatnoca da ce se nakon jednog stanja desiti onset neke druge note ili preci u stanje tisine
    
    T = np.zeros((2*n_notes+1, 2*n_notes+1)) #kreiranje matrice, za pocetak je popunjena nulama

    #Stanje 0 - stanje tisine
    #Stanja 1, 3, 5 ... - stanja onset za svaki ton ponaosob
    #Stanja 2, 4, 6 ... - stanja sustain za svaki ton ponaosob
    #verovatnoca da se iz stanja tisine ponovo predje u stanje tisine
    T[0,0] = p_stay_silence
    for i in range(n_notes):
        T[0, (i*2)+1] = p_ #verovatnoca prelaska iz tisine u onset bilo kog tona
    
    for i in range(n_notes):
        T[(i*2)+1, (i*2)+2] = 1 #verovatnoca prelaska iz onseta u sustain istog tona - uvek je 1

    for i in range(n_notes):
        T[(i*2)+2, 0] = p__ #verovatnoca prelaska iz sustaina u stanje tisine
        T[(i*2)+2, (i*2)+2] = p_stay_note #verovatnoca prelaska iz sustain-a u sustain istog tona
        for j in range(n_notes):        
            T[(i*2)+2, (j*2)+1] = p__ #verovatnoca da ce se preci iz stanja sustaina u onset bilo kog tona 
    #vraca se matrica tranzicije
    return T


#verovatnoce - P[s, t] - verovatnoca da ce se signal naci u stanju s u diskretnom vremenskom trenutku t, na osnovu ulaznog zvucnog signala
def probabilities(y, note_min, note_max, sr, frame_length, window_length, hop_length, pitch_acc, voiced_acc, onset_acc, spread):

    #y - ulazni audio signal
    #note_min - najniza nota obuhvacena programom, u formatu stringa
    #note_max - najvisa nota obuhvacena programom, u formatu stringa
    #sr - sampling rate - u Hz
    #frame_length - duzina frejmova u sample-ovima
    #window_length - frame_length // 2 - duzina "prozora" za procenu korelacije izmedju stanja, u semplovima
    #hop_length - frame_length // 4 - parametar bitan za funkciju pyin i koristi se za Furijeovu transformaciju unutar te funkcije, broj semplova izmedju susednih procena ove funkcije
    #pitch_acc - verovatnoca da je procena visine tona tacna 
    #voiced_acc - verovatnoca da je tacno procenjeno da li frame sadrzi zvuk ili ne
    #onset_acc - verovatnoca da je tacno procenjeno prisustvo onseta
    #spread - verovatnoca da je doslo do devijacije od pola tona, usled "ukrasa" na tonu
    fmin = librosa.note_to_hz(note_min) #frekvencija najnizeg tona
    fmax = librosa.note_to_hz(note_max) #frekvencija najviseg tona
    midi_min = librosa.note_to_midi(note_min) #konvertuje string notaciju u odgovarajuci MIDI "broj" note - u ovom slucaju najnizeg tona
    midi_max = librosa.note_to_midi(note_max) #isto kao prethodna linija, samo se odnosi na najvisi ton
    n_notes = midi_max - midi_min + 1 #ukupan broj tonova obuhvacenih programom

    #f0 - vremeska sekvenca frekvencija osnovnih harmonika koji se pojavljuju u melodiji
    #voiced_flag - niz bool flagova, koji oznacava da li odredjeni frame sadrzi zvuk ili ne
    #voiced_prob - niz verovatnoca da odredjeni frame sadrzi zvuk
    f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin*0.9, fmax*1.1, sr, frame_length, window_length, hop_length)
    #funkcija koja procenjuje devijaciju u stimu - u pitanju je float -0.5 do 0.5
    tuning = librosa.pitch_tuning(f0)
    f0_ = np.round(librosa.hz_to_midi(f0-tuning)).astype(int) #"nastimovacemo" osnovne harmonike
    #kreiramo niz markera koji nam oznacavaju frame-ove koji sadrze onset neke note
    onsets = librosa.onset.onset_detect(y, sr=sr, hop_length=hop_length, backtrack=True) 

    #kreiramo matricu verovatnoca i za pocetak je popunjavamo jedinicama
    #stanja su ista kao sto su opisana u funkciji za generisanje matrice tranzicije
    P = np.ones( (n_notes*2 + 1, len(f0)) )

    for t in range(len(f0)):
        #za stanje tisine
        if voiced_flag[t]==False:
            P[0,t] = voiced_acc #ako nema prisustva zvuka u frame-u, verovatnoca je navedena
        else:
            P[0,t] = 1-voiced_acc #ako ima, verovatnoca je suprotna 

        for j in range(n_notes):
            if t in onsets:
                P[(j*2)+1, t] = onset_acc #verovatnoca za onsete
            else:
                P[(j*2)+1, t] = 1-onset_acc
            #verovatnoca za sustain-e
            if j+midi_min == f0_[t]:
                P[(j*2)+2, t] = pitch_acc #ako se ton poklapa sa nekim od odsviranih, dodeljuje mu se verovatnoca za tacnost procene visine

            elif np.abs(j+midi_min-f0_[t])==1:
                P[(j*2)+2, t] = pitch_acc * spread #ako dodje do devijacije usled "ukrasa" na tonu, dodeljuje se navedena verovatnoca

            else:
                P[(j*2)+2, t] = 1-pitch_acc #u ostalim slucajevima, dodeljuje se suprotna verovatnoca

    return P

#tranksripcija procenjenih stanja u note koje su razumljive MIDI protokolom
def states_to_notes(states, note_min, note_max, hop_time):
    #states - stanja procenjena Viterbi algoritmom
    #note_min - najniza nota obuhvacena programom, u formatu stringa
    #note_max - najvisa nota obuhvacena programom, u formatu stringa
    #hop_time - vremenski interval izmedju dva stanja
    midi_min = librosa.note_to_midi(note_min) #konvertuje string notaciju u odgovarajuci MIDI "broj" note - u ovom slucaju najnizeg tona
    midi_max = librosa.note_to_midi(note_max) #isto kao prethodna linija, samo se odnosi na najvisi ton
    
    states_ = np.hstack( (states, np.zeros(1))) #na kraju niza dodaje se jedna nula da oznaci kraj, odnosno poslednji ofset - horizontal stack

    #tri prethodno navedena stanja
    silence = 0
    onset = 1
    sustain = 2

    my_state = silence #inicijalno postavljamo da je stanje tisina
    output = []
    
    last_onset = 0
    last_offset = 0
    last_midi = 0
    for i in range(len(states_)):
        #ako je trenutno postavljeno stanje tisina
        if my_state == silence:
            #ako se u trenutku desio onset iz stanja tisine
            if int(states_[i]%2) != 0:
                last_onset = i * hop_time #obelezavamo sadasnji trenutak timestamp-om
                last_midi = ((states_[i]-1)/2)+midi_min #nalazimo MIDI broj trenutnog tona
                last_note = librosa.midi_to_note(last_midi) #nalazimo i oznaku ove note u formatu stringa
                my_state = onset #sada postavljamo trenutno stanje u onset

        #ako je trenutno stanje bilo onset
        elif my_state == onset:
            if int(states_[i]%2) == 0:
                my_state = sustain #iz onseta uvek prelazi u sustain

        #ako je trenutno stanje bilo sustain
        elif my_state == sustain:
            if int(states_[i]%2) != 0:
                #ako se preslo u stanje onseta
                last_offset = i*hop_time #obelezavamo ofset poslednje odsvirane note
                my_note = [last_onset, last_offset, last_midi, last_note] #kreiramo niz sa informacijama o poslednjoj odsviranoj i zavrsenoj noti
                output.append(my_note) #i dodajemo ga na izlazni niz
                
                last_onset = i * hop_time #sada setujemo novi onset
                last_midi = ((states_[i]-1)/2)+midi_min #novu MIDI vrednost
                last_note = librosa.midi_to_note(last_midi) #novu oznaku tona
                my_state = onset #i setujemo trenutno stanje na onset

            #ako se stigne do kraja niza stanja
            elif states_[i]==0:
                last_offset = i*hop_time #postavlja se poslednji ofset za taj trenutak
                my_note = [last_onset, last_offset, last_midi, last_note] #kreira se niz sa postojecim informacijama
                output.append(my_note) #dodaje se na izlaz
                my_state = silence #i resetuje stanje na tisinu

    #na kraju vracamo note
    return output


#ova funkcija je namenjena za kreiranje i upis u MIDI fajl
def notes_to_midi(y, notes):

    #y - ulaz wav audio fajla
    #notes - stanja dobijena kao izlaz funkcije states_to_notes
    bpm = librosa.beat.tempo(y)[0] #izracunavanje bpm - beats per minute - koristeci ovu fju
    quarter_note = 60/bpm #trajanje cetvrtine note u sekundama
    #ticks_per_quarter = 1024
    
    onsets = np.array([n[0] for n in notes]) #niz onseta
    offsets = np.array([n[1] for n in notes]) #niz offseta
    
    onsets = onsets / quarter_note #prikazujemo ih kao umnoske cetvrtine note - da bismo znali da procenimo posle koliko nota je pocela sledeca
    offsets = offsets  / quarter_note #slicno i za offsete
    durations = offsets-onsets #i ovde izracunavamo note po trajanju, u vidu umnozaka cetvrtine
    
    #kreiramo MIDI fajl sa jednom trakom
    MyMIDI = midiutil.MIDIFile(1)
    #dodajemo track, time, tempo
    MyMIDI.addTempo(0, 0, bpm)

    #dodaju se note u fajl koriscenjem istog paketa - track, channel, pitch, time + i, duration, volume
    for i in range(len(onsets)):
        MyMIDI.addNote(0, 0, int(notes[i][2]), onsets[i], durations[i], 100)

    return MyMIDI

#funkcija koja poziva sve ostale
def run(file_in, file_out):
    #predefinisani parametri koje cemo koristiti za funkcionisanje programa
    note_min='A2' #zadata najniza nota
    note_max='E6' #zadata najvisa nota
    voiced_acc = 0.9 #verovatnoca da je tacno procenjeno da u nekom frame-u postoji zvuk
    onset_acc = 0.8 #verovatnoca da je tacno procenjen onset
    frame_length=2048 #duzina frejma u semplovima
    window_length=1024 #duzina "prozora" za procenu korelacije izmedju stanja, u semplovima
    hop_length=256 #broj semplova izmedju procena Viterbi algoritma
    pitch_acc = 0.99 #procena tacnosti visina tonova
    spread = 0.6 #procena devijacije 
    
    y, sr = librosa.load(file_in) #koristimo funkciju paketa librosa za ucitavanje wav fajla
    #njegova sadrzina se cuva u promenljivoj y, a sample rate u sr

    #prvo definisemo matricu tranzicije
    T = transition_matrix(note_min, note_max, 0.9, 0.2)
    #zatim matricu verovatnoce
    P = probabilities(y, note_min, note_max, sr, frame_length, window_length, hop_length, pitch_acc, voiced_acc, onset_acc, spread)
    p_init = np.zeros(T.shape[0]) #inicijalna distribucija stanja
    p_init[0] = 1 #inicijalno stanje - stanje tisine
    
    states = librosa.sequence.viterbi(P, T, p_init=p_init) #pozivamo algoritam, koji nam vraca stanja signala
    notes = states_to_notes(states, note_min, note_max, hop_length/sr) #zatim ova stanja prevodimo u note
    MyMIDI = notes_to_midi(y, notes) #a note u MIDI
    #i onda izvodimo write binary nad MIDI fajlom
    with open(file_out, "wb") as output_file:
        MyMIDI.writeFile(output_file)

    
#izvrsavanje
print("Converting files:")
file_in = sys.argv[1] #citanje ulaznog fajla iz konzole
file_out = sys.argv[2] #citanje izlaznog fajla iz konzole
print(sys.argv[1], sys.argv[2])    
run(file_in, file_out) #pokretanje procesa

