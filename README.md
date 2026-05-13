# MultiGrasp Sim-to-Real

## 1. Project Overview

Robotische Hände können prinzipiell eine Vielzahl von Objekten greifen, erfordern aber typischerweise aufwendige manuelle Programmierung für jeden neuen Greiftyp oder jede Objektkategorie. Ziel dieses Projekts ist es, diesen Prozess durch Reinforcement Learning zu automatisieren: Eine Policy soll selbstständig lernen, Objekte unterschiedlicher Form, Größe und Masse zu greifen, ohne dass Greifbewegungen explizit vorgegeben werden.

Trainiert werden zwei spezialisierte Policies, die sich an der menschlichen Greiftaxonomie nach Feix et al. (2014) orientieren. Der Power Grasp zielt darauf ab, größere Objekte kraftvoll mit allen Fingern zu umfassen, während der Precision Grasp kleinere Objekte präzise zwischen Daumen und Fingerspitzen greift. Das gesamte Training findet in einer PyBullet-Simulation statt, da direktes Training auf echter Hardware zu langsam, kostspielig und riskant wäre. Die trainierte Policy wird anschließend auf eine echte AR10-Hand (10 Freiheitsgrade) übertragen, die über einen Sawyer-Roboterarm positioniert und über einen Pololu Maestro Servo-Controller angesteuert wird.

---

## 2. Approach

### 2.1 Menschliche Greifstrategien als Grundlage

Ein klassischer Ansatz beim robotischen Greifen ist es, Kontaktpunkte am Objekt zu definieren und festzulegen welcher Finger wo auf dem Objekt aufsetzen soll. Dieser Ansatz erfordert genaue Kenntnis der Objektgeometrie und lässt sich schlecht auf unbekannte Objekte übertragen.

In diesem Projekt wird ein anderer Weg gewählt, der sich an der Art orientiert, wie Menschen greifen. Nach Feix et al. (2014) lassen sich menschliche Griffe danach unterscheiden, wo das Objekt im Kontakt mit der Hand landet. Beim Power Grasp wird das Objekt gegen die Handfläche gedrückt, beim Precision Grasp gegen den Daumen. Anstatt also zu definieren wo die Finger das Objekt berühren sollen, wird festgelegt wo das Objekt im Verhältnis zur Hand positioniert sein soll.

Umgesetzt wird das durch sogenannte Pregrasp-Posen. Vor jedem Greifversuch wird die Hand so über dem Objekt positioniert, dass das Objekt bereits in der richtigen Zone liegt, also nah an der Handfläche beim Power Grasp oder nah am Daumen beim Precision Grasp. Die Finger sind dabei noch geöffnet. Die RL Policy lernt dann ausschließlich, wie die Finger geschlossen werden müssen, um das Objekt stabil gegen die Hand zu drücken und anzuheben. Die eigentliche Greifplanung ist damit auf ein klar definiertes, lernbares Teilproblem reduziert.

### 2.2 Sim-to-Real Transfer

Da das Training in Simulation stattfindet, entsteht das grundlegende Problem des Sim-to-Real Transfers. Simulation und echte Hardware stimmen nie perfekt überein, da Reibungswerte, Motorcharakteristiken und Objekteigenschaften in der Realität von den Simulationswerten abweichen. Eine Policy die ausschließlich auf fixen Simulationsparametern trainiert wird, würde auf der echten Hardware versagen.

Diesem Problem wird mit zwei Strategien begegnet. Erstens werden durch Domain Randomization die physikalischen Parameter der Simulation pro Episode zufällig variiert, darunter Motorkraft, Fingerkuppenreibung, Objektmasse, Objektgröße und die genaue Startposition der Hand. Die Policy lernt dadurch, robust gegenüber Parametervariation zu sein. Zweitens wird das Observation-Design bewusst auf Signale beschränkt, die zwischen Simulation und echter Hardware konsistent sind. Was das konkret bedeutet und wie dieses Signal aufgebaut ist, wird in Abschnitt 3.1 erläutert.

### 2.3 Software-Architektur

Der Code gliedert sich in vier Module. Das `sim`-Modul enthält die gesamte Trainingsumgebung. Das `training`-Modul steuert den PPO-Trainingsprozess. Das `eval`-Modul deckt die Auswertung in Simulation und auf echter Hardware ab. Das `hardware`-Modul enthält die Schnittstellen zur echten AR10-Hand und zum Sawyer-Arm.

### 2.4 Projektablauf

Das Projekt gliedert sich in zwei Phasen. Zunächst wurden die RL Policies in Simulation trainiert und dort evaluiert. Anschließend wurden die Policies auf der echten AR10-Hand an industriellen Benchmarkobjekten getestet. Um die Hand dabei reproduzierbar in die richtigen Pregrasp-Positionen zu bringen, wurde ein eigenes Positionierungssystem auf Basis des Sawyer-Arms entwickelt, das in Kapitel 6 beschrieben wird.

---

## 3. Environment Design

Die RL-Umgebung definiert, was die Policy wahrnehmen kann (Observation Space), welche Aktionen sie ausführen darf (Action Space) und wie ihre Leistung bewertet wird (Reward).

### 3.1 Observation Space

Damit eine RL Policy sinnvolle Entscheidungen treffen kann, braucht sie Feedback über den aktuellen Zustand der Hand und des Objekts. Da die AR10-Hand keine Berührungssensoren besitzt und eine Kamera den Ansatz unnötig komplex machen würde, wird als Feedback der Tracking-Fehler der Servomotoren verwendet, ein Signal das die Hand selbst liefert. Jeder Servo der AR10 kennt seinen Zielwinkel (q_target) und misst gleichzeitig seinen tatsächlichen Winkel (q_measured). Solange ein Finger sich frei bewegen kann, ist die Differenz zwischen beiden Werten gering. Wenn ein Finger jedoch gegen ein Objekt drückt und blockiert wird, kann er seinen Zielwinkel nicht mehr erreichen, sodass die Differenz ansteigt. Dieses Signal lässt sich sowohl in der Simulation als auch auf der echten Hardware zuverlässig messen.

Da die genaue Servo-Tracking-Charakteristik zwischen Simulation und echter Hardware systematisch abweicht, wird der Tracking-Fehler nicht direkt als kontinuierlicher Wert verwendet. Stattdessen wird er durch einen Schwellwert in ein binäres Kontaktsignal umgewandelt, sodass ein Finger entweder als im Kontakt gilt oder nicht. Dieses binäre Signal ist robust gegenüber den genauen Unterschieden zwischen Simulation und Hardware. Die Idee, Kontakt über einen Tracking-Fehler zu erkennen, orientiert sich an der Forschung von Westling & Johansson (1984), die taktile Signale als wesentliche Grundlage für stabile Griffe identifiziert haben.

Die Observation besteht damit aus zwei Komponenten. Die erste sind die binären Kontaktsignale, ein Bit pro Finger. Die zweite sind die aktuellen Zielwinkel der aktiven Gelenke (q_target, normalisiert auf [0, 1]). Diese zweite Komponente ist notwendig, damit die Policy ihre eigene Handkonfiguration kennt. Ohne sie könnte die Policy nicht unterscheiden ob ein Finger gerade geöffnet oder fast geschlossen ist.

### 3.2 Action Space

Die Policy steuert pro Schritt die Gelenkwinkel der aktiven Finger. Jede Aktion ist ein kontinuierlicher Wert zwischen -1 und 1, der angibt wie stark ein Gelenk in diesem Schritt bewegt werden soll.

Alle Fingergelenke können dabei ausschließlich schließen. Da die Pregrasp-Pose die Finger bereits geöffnet über dem Objekt positioniert, muss die Policy nur noch lernen wie sie die Finger schließt. Eine Öffnungsbewegung wäre kontraproduktiv und würde den Lernprozess unnötig erschweren. Einzige Ausnahme ist die Daumen-Abduktion, also die seitliche Bewegung des Daumens, die bidirektional gesteuert wird um eine feinere Positionierung zu ermöglichen.

Für den Precision Grasp gibt es zusätzlich Winkelgrenzen für die mittleren Fingergelenke. Ohne diese Grenzen könnten die Finger zu weit schließen und kleine Objekte überfahren, anstatt sie zu greifen.

### 3.3 Reward

Der Reward gibt der Policy eine Rückmeldung darüber, wie gut ein Greifversuch war. Er setzt sich aus drei Teilen zusammen.

Pro Schritt erhält die Policy eine kleine Strafe, unabhängig davon was sie tut. Das zwingt die Policy dazu, effizient zu greifen und nicht passiv zu warten.

Zusätzlich gibt es einen Bonus für jeden Finger der Kontakt mit dem Objekt hat. Dieser Bonus ist nach oben begrenzt, damit die Policy nicht lernt möglichst viele Finger nacheinander zu schließen nur um Punkte zu sammeln, sondern einen stabilen Griff mit mehreren Fingern gleichzeitig anstrebt.

Der größte Reward wird am Ende einer Episode vergeben und hängt davon ab, ob der Griff erfolgreich war. Wann genau eine Episode beendet wird und wie der Erfolg gemessen wird, beschreibt die Lift-Logik im nächsten Abschnitt.

### 3.4 Lift-Logik

Die Lift-Logik bestimmt, wann die Policy aufhört zu greifen und ob der Griff als erfolgreich gilt. Sie orientiert sich an der Beschreibung des menschlichen Greifprozesses von Westling & Johansson (1984), der in drei aufeinanderfolgenden Phasen abläuft.

In der ersten Phase schließen die Finger bis Kontakt mit dem Objekt besteht. Kontakt gilt als bestätigt, sobald genug Finger für mehrere aufeinanderfolgende Schritte ein Kontaktsignal liefern. Diese Mehrfachbestätigung verhindert, dass ein einzelner Ausreißer im Signal den Griff vorzeitig auslöst.

In der zweiten Phase, der Stabilisierungsphase, bleiben die Finger in ihrer Position und die Hand bewegt sich noch nicht. Diese Phase gibt der Policy Zeit, die Fingerkonfiguration am Objekt auszurichten bevor der eigentliche Lift beginnt. Weiner et al. (2021) zeigen, dass genau dieser Übergang zwischen Kontaktaufnahme und Kraftaufbau entscheidend für einen stabilen Griff ist.

In der dritten Phase wird die Hand angehoben und geprüft ob das Objekt mitgenommen wird. Wird das Objekt um mindestens 3 cm angehoben, gilt der Griff als erfolgreich und die Policy erhält einen hohen positiven Reward. Fällt das Objekt herunter, gibt es eine Strafe.

---

## 4. Sim-to-Real Transfer

### 4.1 Warum die Servo-Charakteristik in Simulation und Realität abweicht

In Abschnitt 2.2 wurde beschrieben, dass das Observation-Design bewusst auf Signale beschränkt wird, die zwischen Simulation und echter Hardware konsistent sind. Der Grund dafür liegt in einer grundlegenden mechanischen Eigenschaft der AR10-Hand.

Die Firgelli-Linearaktuatoren der AR10 sind nicht rückwärtsantreibbar. Das bedeutet, wenn ein Finger gegen ein Objekt drückt, kann das Objekt den Finger nicht mechanisch zurückdrücken. In der Simulation hingegen ist ein Gelenk in beide Richtungen beweglich, sodass externe Kräfte es zurückdrücken können und der simulierte Motor dagegen anarbeiten muss. Dadurch entsteht in der Simulation ein anderes Tracking-Fehlersignal als in der Realität, selbst wenn das Objekt und die Kraft identisch wären.

Das macht es unmöglich, den kontinuierlichen Tracking-Fehler direkt als Observation zu verwenden, da der gelernte Zusammenhang zwischen Tracking-Fehler und Kontaktzustand in Simulation und Realität nicht übereinstimmt. Die binäre Schwellwertierung aus Abschnitt 3.1 löst dieses Problem, indem sie nur die Information "Kontakt vorhanden oder nicht" weitergibt und die genauen Zahlenwerte ignoriert.

### 4.2 Threshold-Kalibrierung

Da der Schwellwert der Kontakterkennung in Simulation bestimmt wird, muss er nach dem Training einmalig auf der echten Hardware kalibriert werden. Dazu wird die Hand in einer kontrollierten Sequenz gegen bekannte Objekte gedrückt und der Tracking-Fehler der einzelnen Gelenke aufgezeichnet. Aus diesen Messungen wird ein Schwellwert abgeleitet, der auf der echten Hand zuverlässig zwischen freier Bewegung und Kontakt unterscheidet. Dieser kalibrierte Schwellwert ersetzt beim Deployment auf echter Hardware den Simulationswert.


---

## 5. Training Setup

### 5.1 Algorithmus

Als Lernalgorithmus wird Proximal Policy Optimization (PPO) verwendet. PPO gehört zur Klasse der on-policy RL-Algorithmen und begrenzt durch einen Clipping-Mechanismus wie stark sich die Policy pro Update-Schritt verändern darf. Das verhindert, dass ein einzelnes schlechtes Batch die Policy in eine schlechte Richtung treibt aus der sie sich nicht mehr erholen kann. Für dieses Projekt ist das besonders relevant, da der größte Reward erst am Ende einer Episode nach dem Lift-Test vergeben wird. Frühzeitige Policy-Zusammenbrüche durch zu große Update-Schritte würden das Training dauerhaft destabilisieren.

### 5.2 Trainingsinfrastruktur

Das Training läuft mit mehreren parallelen Simulationen gleichzeitig, eine pro CPU-Kern. Dadurch werden pro Zeiteinheit deutlich mehr Trainingsdaten gesammelt als mit einer einzelnen Umgebung.

### 5.3 Physik-Parameter

Die Simulationsfrequenz beträgt 240 Hz, mit 5 Physik-Schritten pro Policy-Step. Daraus ergibt sich eine Kontrollfrequenz von 48 Hz. Dieser Wert orientiert sich an der echten Hardware, da der Pololu Maestro Servo-Controller der AR10-Hand ein Standard-PWM-Signal mit ~50 Hz erzeugt. Die Kontrollfrequenz in der Simulation approximiert damit die tatsächliche Reaktionsrate der echten Hardware, sodass die zeitliche Dynamik des Greifens in Simulation und Realität vergleichbar bleibt.


---

## 6. Pregrasp Positioning System

Bevor die RL Policy auf echter Hardware ausgeführt wird, muss die AR10-Hand präzise in die Pregrasp-Pose gebracht werden. Der Sawyer-Roboterarm übernimmt diese Aufgabe, indem er die AR10-Hand hält und sie vor jedem Greifversuch an die richtige Position über dem Objekt bewegt. Dafür muss für jedes Benchmarkobjekt vorab bestimmt werden, in welche Gelenkwinkelkonfiguration der Sawyer fahren muss.

Um diese Konfigurationen zu berechnen, wird das reale Setup exakt in Simulation nachgebaut. Sawyer-Arm und Benchmarkobjekt befinden sich dabei im selben räumlichen Verhältnis zueinander wie in der Realität. Da sowohl der Sawyer als auch die AR10-Hand als URDF-Modelle vorliegen, kann die gesamte Berechnung in Simulation stattfinden, ohne die echte Hardware dafür zu benötigen.

Der Berechnungsablauf ist folgender. Zunächst wird in Simulation bestimmt, wo die AR10-Hand relativ zum Objekt positioniert sein muss, also die Pregrasp-Pose. Anschließend wird per Inverser Kinematik berechnet, welche Gelenkwinkel der Sawyer einnehmen muss um die Hand genau dorthin zu bringen. Die so ermittelten Gelenkwinkel werden direkt auf den echten Sawyer übertragen, der die Hand dann in die entsprechende Position fährt.

Die Ergebnisse dieser Berechnungen werden in einer Lookup-Table gespeichert, eine Eintragszeile pro Benchmarkobjekt. Beim Deployment muss keine Berechnung mehr stattfinden. Die Hand wird anhand des erkannten Objekts einfach in die passende vorberechnete Position gefahren.

---

## 7. Results

Die Policies werden auf drei Arten evaluiert. Zunächst wird in Simulation ein breites Spektrum an Objekten systematisch getestet, um ein allgemeines Bild der Generalisierungsfähigkeit zu bekommen. Danach werden die Policies in Simulation auf den realen Benchmarkteilen getestet, also auf denselben Objekten die später auch auf echter Hardware verwendet werden. Abschließend werden die Policies auf der echten AR10-Hand getestet.

### 7.1 Objektgrid in Simulation

Um zu verstehen welche Objekte die Policy gut oder schlecht greifen kann, wird ein Grid aller möglichen Objektformen und -größen systematisch durchlaufen. Da die Domain Randomization aktiv bleibt, wird jedes Objekt mehrfach getestet um zufällige Variationen auszumitteln. Das Ergebnis zeigt die Erfolgsrate aufgeschlüsselt nach Objektkategorie und -größe.

#### 7.1.1 Power Grasp

*[Wird nach Abschluss des Trainings ausgefüllt]*

#### 7.1.2 Precision Grasp

*[Wird nach Abschluss des Trainings ausgefüllt]*

### 7.2 Benchmarkteile in Simulation

Die Policies werden auf den ungesehenen realen Benchmarkteilen in Simulation getestet. Da diese Objekte nicht Teil des Trainings waren, gibt dieser Test Aufschluss darüber wie gut die Policy auf bekannte reale Zielobjekte generalisiert.

#### 7.2.1 Power Grasp

*[Wird nach Abschluss des Trainings ausgefüllt]*

#### 7.2.2 Precision Grasp

*[Wird nach Abschluss des Trainings ausgefüllt]*

### 7.3 Echte Hardware

Die Policies werden mit der echten AR10-Hand an den realen Benchmarkteilen getestet. Der Vergleich mit den Simulationsergebnissen aus 7.2 zeigt direkt wie groß die Sim-to-Real Lücke in der Praxis ist.

#### 7.3.1 Power Grasp

*[Wird nach Abschluss des Trainings ausgefüllt]*

#### 7.3.2 Precision Grasp

*[Wird nach Abschluss des Trainings ausgefüllt]*

---

## 8. Discussion & Limitations

