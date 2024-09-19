import os
import gradio as gr
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 한글 폰트 설정 (NanumGothic 사용)
plt.rcParams['font.family'] = 'NanumGothic'

# 모음별 포먼트 범위 설정
formant_ranges_KOR_man = {
    'ㅏ': {'F1': (600, 850), 'F2': (1100, 1250), 'color': 'red'},
    'ㅣ': {'F1': (200, 300), 'F2': (1700, 2200), 'color': 'blue'},
    'ㅜ': {'F1': (300, 450), 'F2': (590, 900), 'color': 'green'},
    'ㅗ': {'F1': (300, 450), 'F2': (580, 820), 'color': 'purple'},
    'ㅡ': {'F1': (300, 500), 'F2': (1200, 1500), 'color': 'orange'},
    'ㅓ': {'F1': (430, 650), 'F2': (800, 1300), 'color': 'brown'},
    'ㅐ': {'F1': (400, 520), 'F2': (1700, 1900), 'color': 'pink'}
}

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

# 모음 사각도를 그리는 함수 (타원형으로 기준 구간 표시)
def plot_vowel_space(f1, f2, vowel):
    plt.figure(figsize=(8, 8))

    # 기준 모음 위치 (타원형으로 표시)
    for v, formant in formant_ranges_KOR_man.items():
        if 'F1' in formant and 'F2' in formant:
            f1_range = formant['F1']
            f2_range = formant['F2']
            ellipse = Ellipse(xy=((f2_range[0] + f2_range[1]) / 2, (f1_range[0] + f1_range[1]) / 2),
                              width=f2_range[1] - f2_range[0],
                              height=f1_range[1] - f1_range[0],
                              edgecolor=formant['color'], facecolor=formant['color'], lw=2, alpha=0.3, label=f'기준 {v}')
            plt.gca().add_patch(ellipse)

    # 실제 사용자의 포먼트 값 (검은 점, 노란색 테두리)
    if f1 is not None and f2 is not None:
        plt.scatter(f2, f1, label=f"실제 {vowel}", color='black', s=500, edgecolors='yellow', linewidth=2)

    plt.xlabel('F2 (Hz)')
    plt.ylabel('F1 (Hz)')
    plt.title(f'모음 사각도: {vowel}')
    plt.gca().invert_yaxis()  # F1 값이 높을수록 아래로 가게끔 (음성학적 관례)

    # 범례 간격 조절
    plt.legend(loc='best', frameon=True, borderpad=1.5, labelspacing=1.2)

    plt.grid(True)
    plt.savefig("vowel_space.png")
    plt.close()
    
# 메인 분석 함수
def analyze_audio(consent, audio, vowel):
    if '아니요' in consent:
        return "개인정보 수집에 동의하지 않으면 서비스를 이용할 수 없습니다.", None

    if audio is None:
        return "오디오 파일이 없습니다.", None

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

    return "분석 완료", "vowel_space.png"

# Gradio UI 설정
def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("Korhub2.0 음성분석기")

        # 개인정보 수집 동의
        consent = gr.CheckboxGroup(['예', '아니요'], label='개인정보 수집에 동의하십니까?')
        
        vowel = gr.Dropdown(list(formant_ranges_KOR_man.keys()), label="분석할 모음을 선택하세요")

        audio_input = gr.Audio(type="filepath", label="음성을 녹음하세요")
        submit_btn = gr.Button("제출")

        result = gr.Textbox(label="결과")
        formant_result = gr.Image(label="모음 사각도")

        # 개인정보 동의 및 분석 처리
        submit_btn.click(fn=analyze_audio, 
                         inputs=[consent, audio_input, vowel], 
                         outputs=[result, formant_result])

    return demo

# Gradio 앱 실행
import os

# Use the 'PORT' environment variable from Koyeb
port = int(os.getenv('PORT', 8000))  # Default to 8000 if 'PORT' is not set
demo.launch(share=True, server_port=port)
