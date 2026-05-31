const rcrd_btn = document.getElementById('recrdBtn')
const audio_player = document.getElementById('audioPlayer')
const down_btn = document.getElementById('down_btn')
const down_wrap = document.getElementById('download_wrap')

down_wrap.hidden = true

let isRecording = false
let generating = false
let generated = false
let mediaRecorder = null
let chunks = []
rcrd_btn.addEventListener('click', async function () {
    if (generated) {
            generated = false
            generating = false
            chunks = []
            mediaRecorder = null
            audio_player.src = ''
            audio_player.hidden = true
            down_wrap.hidden = true
            rcrd_btn.textContent = 'Record'
            rcrd_btn.style.color = '#2ded97'
    }
    else if (!generated) {
        if (!isRecording) {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
            mediaRecorder = new MediaRecorder(stream)
            mediaRecorder.ondataavailable = (e) => chunks.push(e.data)
            mediaRecorder.start()
            isRecording = true
            rcrd_btn.textContent = 'Stop'
            rcrd_btn.style.color = 'red'
        } else if (!generating) {
            generating = true
            rcrd_btn.textContent = 'processing'
            rcrd_btn.style.color = '#f0cd22'
            isRecording = false
            
            rcrd_btn.disabled = true
            mediaRecorder.onstop = async () => {
                const blob = new Blob(chunks, { type: 'audio/wav' })
                const formData = new FormData()
                formData.append('recording', blob, 'recording.wav')

                const response = await fetch('/generate', {
                    method: 'POST',
                    body: formData
                })

                const audioBlob = await response.blob()
                const audioUrl = URL.createObjectURL(audioBlob)
                audio_player.src = audioUrl
                audio_player.hidden = false
                down_wrap.hidden = false
                down_btn.href = audioUrl
                down_btn.download = 'generated_melody.wav'
                generated = true
                // now enable button
                rcrd_btn.disabled = false
                rcrd_btn.textContent = 'Record Again'
                rcrd_btn.style.color = '#2ded97'
            }
            mediaRecorder.stop()
            

        }
    }
})