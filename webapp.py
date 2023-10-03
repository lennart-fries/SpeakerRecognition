import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime
from network import recognition
from network.model import NeuralNet, load
import manualCount
import time

# Set Up Page and Assets
st.set_page_config(page_title='Remote Meeting, but fair?!', page_icon=':notebook:', layout='wide')
df = pd.DataFrame(columns=['second', 'speaker'])

# Setup Network and Live Recognition
network = NeuralNet(True, 4, 1)
model_path = ['models/film_speaker_1s.model', 'models/film_gpu_96d3.model', 'models/film_cpu_94d9.model']
model = load(model=network, path=model_path[1])
rec = recognition.LiveRecognition(model, speaker=True, duration=1)

# Sidebar
# Defines the Condition and thus the display of things, when we get to the active part of the study
sidebar = st.sidebar.selectbox('Select Condition:', ('I:MaxFunc', 'II:TextOnly', 'III:Ctrl'))

# TODO:
#   !!! ADD STUDY LINK AND FINALIZE TESTING EVERYTHING
#   !!! Fragebogen (moralisch? inkonsequent - inkonsistent? -- Beschriftung)

# Header
# Introduces the study
with st.container():
    if sidebar == 'I:MaxFunc':
        st.title('Kooperative Moderation')
    elif sidebar == 'II:TextOnly':
        st.title('Moderation mit KI')
    else:
        st.title('Verbesserungen durch KI')

# Text
# Explains the workflow of the study, including the data processing explanation (Can we use a form for that or paper?)
# Links the organizational workflow (Survey Link, etc..) and explains the to do
# Maybe add checkbox here, when done so the lower part only shows after?
with st.container():
    st.subheader('Worum geht es in dieser Studie?')
    st.write('---')
    left_col, right_col = st.columns(2)
    with left_col:
        st.write('In der Pandemie mussten viele verschiedene Firmen und auch staatliche Institutionen neue '
                 'Kommunikationsmöglichkeiten erschließen...')
        st.write('Videokonferenzen waren häufig die Technologie der Wahl, um eine gute Kommunikation zu gewährleisten')
        if sidebar != 'III:Ctrl':
            st.write('Allerdings gab es dabei auch einige Probleme. Mit dieser Studie soll eine mögliche Lösung '
                     'getestet werden')
            st.write('Getestet wird eine Möglichkeit, die Moderation von Videokonferenzen zu verbessern.')
        else:
            st.write('Wir versuchen Möglichkeiten zu finden, mit Künstlicher Intelligenz diese Konferenzen'
                     'zu verbessern')

with st.container():
    st.subheader('Erste Schritte')
    st.write('---')
    st.write('Dein Versuchsleiter sollte dir schon die Einverständniserklärung und das Infoblatt gegeben haben.'
                 ' Ist das nicht der Fall, dann melde dich bitte deswegen.')
    st.write('Zuerst möchten wir dich bitten, folgende Umfrage auszufüllen, bis du aufgefordert wirst, '
                 'dich an den Versuchsleiter zu wenden')
    st.write('[Link zur SoSci Umfrage](https://ls1.psychologie.uni-wuerzburg.de/so/mod-ki/)')

with st.container():
    st.subheader('Studie')
    st.write('---')
    st.write('Vielen Dank dafür, dass du die Umfrage ausgefüllt hast!')
    st.write('Dein Versuchsleiter hat dir erklärt, was im folgenden passiert:')
    st.write('Zuerst startest du die KI, indem du auf den Button unten drückst. '
             'Bitte klicke nur einmal auf den Button.')
    st.write('Danach startest du das Video einer Diskussion auf dem anderen Bildschirm.')
    st.write('Du schlüpfst in die Rolle des Moderators und wirst dabei von der KI unterstützt. '
             'Halte gerne ab und zu das Video an und gebe Feedback, wie du jetzt handeln würdest und warum.')
    st.write('Im Anschluss wirst du von deinem Versuchsleiter noch einige Fragen gestellt bekommen und den Fragebogen '
             'fertig ausfüllen.')
    st.write('Viel Spaß bei der Studie!')

    loop = False
    start = st.button('Starte die KI!')
    if start:
        loop = True
        st.error("Die KI läuft! Bitte diesen Button **NICHT** erneut drücken!")

    placeholder = st.empty()

    if loop:
        manSec = 0
        for seconds in range(2400):
            speaker = rec.classify_live().item()

            df = df.append({'second': seconds, 'speaker': speaker}, ignore_index=True)
            data = [
                ['Person P1', df[df['speaker'] == 3].count()[0]],
                ['Person P2', df[df['speaker'] == 2].count()[0]],
                ['Person P3', df[df['speaker'] == 4].count()[0]],
                ['Person P4', df[df['speaker'] == 1].count()[0]]
            ]

            speaking_time = pd.DataFrame(data, columns=['name', 'seconds'])
            #speaking_time = manualCount.manual_add(manSec)
            #manSec += 1
            with placeholder.container():
                col1, col2 = st.columns(2)
                if sidebar == 'I:MaxFunc':
                    with col1:
                        st.markdown("### First Chart")
                        fig = px.bar(speaking_time, y='name', x='seconds', color="name",
                                     color_discrete_sequence=px.colors.qualitative.Safe, text_auto=True)
                        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
                        st.write(fig)
                    with col2:
                        st.markdown("### Second Chart")
                        fig2 = px.pie(speaking_time, values='seconds', names='name', hole=0.3, color="name",
                                      color_discrete_sequence=px.colors.qualitative.Safe)
                        fig2.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
                        st.write(fig2)
                if sidebar != 'III:Ctrl':
                        st.markdown("### Most recent live suggestion")
                        # Total Seconds
                        if speaking_time["seconds"].sum() == 35:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests:'     
                                        f'Try to get P4 to participate')
                        if speaking_time["seconds"].sum() == 65:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests: '
                                        f'Encourage P1 and P4 to speak more')
                        if speaking_time["seconds"].sum() == 75:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests: '
                                        f'Have P2 and P3 speak less frequently')
                        if speaking_time["seconds"].sum() == 170:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests: '
                                        f'Ask P2 and P3 to leave time for P1 to elaborate')
                        if speaking_time["seconds"].sum() == 220:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests: '
                                        f'Hold P3 back from dominating the discussion')
                        if speaking_time["seconds"].sum() == 240:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests: '
                                        f'Let P1 share their opinion more')
                        if speaking_time["seconds"].sum() == 310:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests: '
                                        f'Suppress monologues, while involving P1 and P4 more')
                        if speaking_time["seconds"].sum() == 360:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests: '
                                        f'Involve P1 more actively into the discussion')
                        if speaking_time["seconds"].sum() == 420:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests: '
                                        f'Make P1 participate by asking them to share their opinion')
                        if speaking_time["seconds"].sum() == 450:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests: '
                                        f'Once more, try more actively to involve P1')
                        if speaking_time["seconds"].sum() == 460:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests: '
                                        f'Remind P2 and P3 to not interrupt mid-sentence')
                        if speaking_time["seconds"].sum() == 490:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests: '
                                        f'Reduce the amount of remaining speaking time for P2, P3 and P4')
                        if speaking_time["seconds"].sum() == 510:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests: '
                                        f'P1 has attempted multiple times to share their opinion, '
                                        f'have the others let them finish their argument')
                        if speaking_time["seconds"].sum() == 550:
                            st.markdown(f'#### {datetime.now().strftime("%H:%M:%S")} -- AI suggests: '
                                        f'Try to steer the remainder of the discussion with a focus on P1 in mind')

                if sidebar == 'III:Ctrl':
                    with col1:
                        st.success("#### This is a placeholder for the AI's visual live representation "
                                   "of speaking time")
                    with col2:
                        st.info("#### This is a placeholder for the AI's textual suggestions "
                                "based on the live situation")

                    st.warning('#### This is some free space for other functionalities, do you have any ideas?')

