import gradio as gr
import whisper
import parselmouth
import numpy as np
import matplotlib.pyplot as plt

# Whisper 모델 불러오기
model = whisper.load_model("tiny")

# 한글 폰트 설정 (NanumGothic 사용)
plt.rcParams['font.family'] = 'NanumGothic'

# 모음별 포먼트 범위 설정
formant_ranges_KOR_man = {
    'ㅏ': {'F1': (600, 850), 'F2': (1100, 1250),'color':'red'},
    'ㅣ': {'F1': (200, 300), 'F2': (1700, 2200), 'color':'blue'},
    'ㅜ': {'F1': (300, 450), 'F2': (590, 900), 'color':'green'},
    'ㅗ': {'F1': (300, 450), 'F2': (580, 820), 'color':'purple'},
    'ㅡ': {'F1': (300, 500), 'F2': (1200, 1500), 'color':'orange'},
    'ㅓ': {'F1': (430, 650), 'F2': (800, 1300), 'color':'brown'},
    'ㅐ': {'F1': (400, 520), 'F2': (1700, 1900), 'color':'pink'}
}

formant_ranges_KOR_woman = {
    'ㅏ': {'F1': (800, 950), 'F2': (1300, 1800), 'color':'red'},
    'ㅣ': {'F1': (200, 350), 'F2': (1200, 2700), 'color':'blue'},
    'ㅜ': {'F1': (330, 440), 'F2': (800, 900), 'color':'green'},
    'ㅗ': {'F1': (360, 460), 'F2': (700, 900), 'color':'purple'},
    'ㅡ': {'F1': (380, 450), 'F2': (1500, 1800), 'color':'orange'},
    'ㅓ': {'F1': (570, 700), 'F2': (950, 1250), 'color':'brown'},
    'ㅐ': {'F1': (540, 650), 'F2': (2200, 2500), 'color':'pink'}
}

formant_ranges_CN_man = {
    'ㅏ': {'F1': (800, 870), 'F2': (1300, 1600), 'color':'red'},
    'ㅣ': {'F1': (350, 410), 'F2': (2400, 2900), 'color':'blue'},
    'ㅜ': {'F1': (400, 490), 'F2': (800, 900), 'color':'green'},
    'ㅗ': {'F1': (500, 560), 'F2': (700, 900), 'color':'purple'},
    'ㅡ': {'F1': (460, 540), 'F2': (1500, 1800), 'color':'orange'},
    'ㅓ': {'F1': (600, 650), 'F2': (950, 1250), 'color':'brown'},
    'ㅐ': {'F1': (400, 580), 'F2': (1800, 2500), 'color':'pink'}
}

formant_ranges_CN_woman = {
    'ㅏ': {'F1': (980, 1290), 'F2': (1350, 1560), 'color':'red'},
    'ㅣ': {'F1': (350, 370), 'F2': (2200, 2800), 'color':'blue'},
    'ㅜ': {'F1': (440, 500), 'F2': (910, 1220), 'color':'green'},
    'ㅗ': {'F1': (500, 570), 'F2': (910, 1100), 'color':'purple'},
    'ㅡ': {'F1': (480, 680), 'F2': (1500, 1540), 'color':'orange'},
    'ㅓ': {'F1': (570, 710), 'F2': (950, 1250), 'color':'brown'},
    'ㅐ': {'F1': (670, 750), 'F2': (2200, 2230), 'color':'pink'}
}

# STT 처리 함수 (한국어 지원 추가)
def transcribe_audio(audio):
    result = model.transcribe(audio, language="ko")  # 한국어 설정 추가
    return result['text']

# 에너지 측정 함수 (에너지가 발생한 구간 찾기)
def detect_energy(sound):
    intensity = sound.to_intensity()
    energy_times = intensity.xs()
    energy_values = intensity.values.T

    # 에너지가 가장 높은 시점 찾기
    max_energy_index = np.argmax(energy_values)
    max_energy_time = energy_times[max_energy_index]

    return max_energy_time

# 포먼트 추출 함수 (에너지가 발생한 구간 기준 0.01초 간격)
def extract_formants_near_energy(sound, energy_time, time_window=0.01):
    formant = sound.to_formant_burg()
    time_step = 0.01  # 1/100초 간격
    start_time = max(0, energy_time - time_window)
    end_time = energy_time + time_window
    formant_values = []

    # 에너지 발생 구간의 앞뒤 0.01초에서 F1, F2 값 추출
    for t in np.arange(start_time, end_time + time_step, time_step):
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        formant_values.append((t, f1, f2))

    return formant_values

# F1, F2 값 확인 및 출력 함수
def print_formants(formant_values):
    print("Time\tF1\tF2")
    for t, f1, f2 in formant_values:
        if f1 and f2:  # F1, F2 값이 존재할 때만 출력
            print(f"{t:.3f}\t{f1:.2f}\t{f2:.2f}")

# 모음 사각도를 그리는 함수 (기준값과 사용자의 실제 값 비교)
def plot_vowel_space(f1, f2, vowel):
    plt.figure(figsize=(8, 8))

    # 기준 모음 위치 (모든 기준 모음, 각각의 색으로 표시)
    for v, formant in formant_ranges_KOR_man.items():
        plt.scatter(formant['F2'], formant['F1'], label=f"기준 {v}", color=formant['color'], s=200)  # 모음별 다른 색상 적용
    
    # 실제 사용자의 포먼트 값 (검은 점, 노란색 테두리)
    if f1 is not None and f2 is not None:
        plt.scatter(f2, f1, label=f"실제 {vowel}", color='black', s=500, edgecolors='yellow', linewidth=2)  # 점 크기 및 테두리
    
    plt.xlabel('F2 (Hz)')
    plt.ylabel('F1 (Hz)')
    plt.title(f'모음 사각도: {vowel}')
    plt.gca().invert_yaxis()  # F1 값이 높을수록 아래로 가게끔 (음성학적 관례)
    plt.legend()
    plt.grid(True)
    plt.savefig("vowel_space.png")
    plt.close()

# 메인 분석 함수
def analyze_audio(consent, audio, vowel, gender):
    if '아니요' in consent:
        return "개인정보 수집에 동의하지 않으면 서비스를 이용할 수 없습니다.", None

    if audio is None:
        return "오디오 파일이 없습니다.", None

    # STT 처리
    transcription = transcribe_audio(audio)

    # Temp 파일에 저장 후 에너지가 발생한 구간 확인
    sound = parselmouth.Sound(audio)
    energy_time = detect_energy(sound)

    # 에너지 발생 구간 기준 1/100초 간격으로 F1, F2 추출
    formant_values = extract_formants_near_energy(sound, energy_time)
    
    # F1, F2 값 출력 (확인용)
    print_formants(formant_values)

    # 평균 F1, F2 계산
    f1_values = [f1 for _, f1, _ in formant_values if f1 is not None]
    f2_values = [f2 for _, _, f2 in formant_values if f2 is not None]
    if not f1_values or not f2_values:
        return "포먼트를 필터링할 수 없습니다.", None

    avg_f1 = np.mean(f1_values)
    avg_f2 = np.mean(f2_values)

    # 모음 사각도 시각화 (기준값과 비교)
    plot_vowel_space(avg_f1, avg_f2, vowel)

    return transcription, "vowel_space.png"

# Gradio UI 설정
def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("Korhub2.0 음성분석기")

        # 개인정보 수집 동의
        consent = gr.CheckboxGroup(['예', '아니요'], label='개인정보 수집에 동의하십니까?')
        gender = gr.Dropdown(["남", "여"], label="성별을 선택하세요")
        nationality = gr.Dropdown('국적을 선택하세요', ['대한민국','베트남','중국','기타'])
        age_group = gr.Dropdown('연령을 선택하세요',['아동(변성기 이전','청소년','청년','장년'])
        
        vowel = gr.Dropdown(list(formant_ranges_KOR_man.keys()), label="분석할 모음을 선택하세요")

        audio_input = gr.Audio(type="filepath", label="음성을 녹음하세요")
        submit_btn = gr.Button("제출")

        transcription = gr.Textbox(label="STT 결과")
        formant_result = gr.Image(label="모음 사각도")

        # 개인정보 동의 및 분석 처리
        submit_btn.click(fn=analyze_audio, 
                         inputs=[consent, audio_input, vowel, gender], 
                         outputs=[transcription, formant_result])

    return demo

# Gradio 앱 실행
demo = create_ui()
demo.launch(share=True)
