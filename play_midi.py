import mido
import click
import time

@click.command()
@click.option('--input', '-i', required=True, help="재생할 MIDI 파일 경로")
def main(input):
    """
    MIDI 파일을 시스템 기본 신디사이저(Windows 내장 MIDI)를 통해 오디오로 재생합니다.
    (추가적인 사운드폰트 설치가 필요 없습니다.)
    """
    try:
        # 사용 가능한 출력 포트 확인
        outputs = mido.get_output_names()
        if not outputs:
            print("[오류] 사용 가능한 MIDI 출력 장치를 찾을 수 없습니다.")
            return
            
        port_name = outputs[0]  # 보통 Windows에서는 'Microsoft GS Wavetable Synth 0'
        print(f"\n🎵 [재생 시작] 파일: {input}")
        print(f"🔊 [출력 장치] {port_name}")
        print("정지하려면 Ctrl+C를 누르세요.\n")
        
        mid = mido.MidiFile(input)
        
        with mido.open_output(port_name) as port:
            for msg in mid.play():
                if not msg.is_meta:
                    port.send(msg)
                
        print("✅ [재생 완료]")
        
    except KeyboardInterrupt:
        print("\n⏹️ [재생 중지됨]")
    except Exception as e:
        print(f"\n❌ 재생 중 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    main()
