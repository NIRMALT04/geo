import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX, 
  User,
  Bot,
  Play,
  Pause,
  Settings
} from 'lucide-react'
import { useEODataStore } from '../stores/eoDataStore'

const VoiceInterface: React.FC = () => {
  const {
    activePersona,
    availablePersonas,
    setActivePersona,
    isListening,
    setIsListening,
    isProcessing,
    setIsProcessing
  } = useEODataStore()

  const [isSpeaking, setIsSpeaking] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [conversation, setConversation] = useState<Array<{
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: string
    audioUrl?: string
  }>>([])

  const audioRef = useRef<HTMLAudioElement>(null)
  const recognitionRef = useRef<any>(null)

  // Initialize speech recognition
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
      recognitionRef.current = new SpeechRecognition()
      
      recognitionRef.current.continuous = true
      recognitionRef.current.interimResults = true
      recognitionRef.current.lang = 'en-US'

      recognitionRef.current.onresult = (event: any) => {
        let finalTranscript = ''
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript
          }
        }
        if (finalTranscript) {
          setTranscript(finalTranscript)
          handleVoiceQuery(finalTranscript)
        }
      }

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error)
        setIsListening(false)
      }

      recognitionRef.current.onend = () => {
        setIsListening(false)
      }
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
    }
  }, [setIsListening])

  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      setIsListening(true)
      setTranscript('')
      recognitionRef.current.start()
    }
  }

  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop()
      setIsListening(false)
    }
  }

  const handleVoiceQuery = async (query: string) => {
    if (!activePersona || !query.trim()) return

    setIsProcessing(true)
    
    // Add user message to conversation
    const userMessage = {
      id: `user_${Date.now()}`,
      role: 'user' as const,
      content: query,
      timestamp: new Date().toISOString()
    }
    
    setConversation(prev => [...prev, userMessage])

    try {
      // Simulate AI processing
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      // Mock AI response based on persona
      const responses = {
        'geo-expert': `Based on the satellite imagery analysis, ${query.toLowerCase().includes('urban') ? 'I can see significant urban development patterns with high built-up density and mixed land use zones.' : 'the spectral signatures indicate diverse land cover types with varying vegetation indices.'}`,
        'eco-analyst': `From an environmental perspective, ${query.toLowerCase().includes('vegetation') ? 'the vegetation health shows moderate NDVI values, suggesting seasonal variations in plant growth.' : 'the ecosystem indicators reveal interesting patterns in biodiversity and habitat distribution.'}`,
        'change-detector': `Analyzing temporal changes, ${query.toLowerCase().includes('change') ? 'I detect significant land use modifications over the selected time period with notable urbanization trends.' : 'the time-series data shows stability in most areas with some localized variations.'}`
      }

      const aiResponse = {
        id: `ai_${Date.now()}`,
        role: 'assistant' as const,
        content: responses[activePersona.id as keyof typeof responses] || 'I can help you analyze satellite imagery and provide insights about Earth observation data.',
        timestamp: new Date().toISOString(),
        audioUrl: '/mock-audio/response.mp3' // Mock audio URL
      }

      setConversation(prev => [...prev, aiResponse])
      
      // Simulate text-to-speech
      setIsSpeaking(true)
      setTimeout(() => setIsSpeaking(false), 3000)

    } catch (error) {
      console.error('Voice query processing error:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  const toggleSpeaking = () => {
    setIsSpeaking(!isSpeaking)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Voice Interface</h1>
          <p className="text-slate-400">Conversational AI for spatial queries</p>
        </div>

        {/* Persona Selection */}
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-white mb-3">Select AI Persona</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {availablePersonas.map((persona) => (
              <button
                key={persona.id}
                onClick={() => setActivePersona(persona)}
                className={`p-4 rounded-lg border transition-all text-left ${
                  activePersona?.id === persona.id
                    ? 'bg-blue-500/20 border-blue-500/50'
                    : 'bg-slate-800/50 border-slate-600/50 hover:bg-slate-700/50'
                }`}
              >
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                    <Bot className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">{persona.name}</h3>
                    <p className="text-xs text-slate-400">{persona.personality.tone}</p>
                  </div>
                </div>
                <p className="text-sm text-slate-300">{persona.description}</p>
                <div className="flex flex-wrap gap-1 mt-2">
                  {persona.personality.expertise.slice(0, 2).map((skill) => (
                    <span
                      key={skill}
                      className="px-2 py-1 bg-slate-600/50 text-xs text-slate-300 rounded"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </button>
            ))}
          </div>
        </div>

        {activePersona && (
          <>
            {/* Voice Controls */}
            <div className="mb-6 bg-slate-800/50 backdrop-blur-sm border border-slate-600/50 rounded-lg p-6">
              <div className="flex items-center justify-center gap-6">
                {/* Listening Button */}
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={isListening ? stopListening : startListening}
                  disabled={isProcessing}
                  className={`w-20 h-20 rounded-full flex items-center justify-center transition-all ${
                    isListening
                      ? 'bg-red-500 hover:bg-red-600 shadow-lg shadow-red-500/30'
                      : 'bg-blue-500 hover:bg-blue-600 shadow-lg shadow-blue-500/30'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  {isListening ? (
                    <MicOff className="w-8 h-8 text-white" />
                  ) : (
                    <Mic className="w-8 h-8 text-white" />
                  )}
                </motion.button>

                {/* Speaking Button */}
                <button
                  onClick={toggleSpeaking}
                  disabled={!conversation.length}
                  className={`p-4 rounded-full transition-all ${
                    isSpeaking
                      ? 'bg-green-500 hover:bg-green-600'
                      : 'bg-slate-600 hover:bg-slate-500'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  {isSpeaking ? (
                    <VolumeX className="w-6 h-6 text-white" />
                  ) : (
                    <Volume2 className="w-6 h-6 text-white" />
                  )}
                </button>

                {/* Settings */}
                <button className="p-4 bg-slate-600 hover:bg-slate-500 rounded-full transition-colors">
                  <Settings className="w-6 h-6 text-white" />
                </button>
              </div>

              {/* Status */}
              <div className="text-center mt-4">
                {isListening && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-red-400 font-medium"
                  >
                    Listening...
                  </motion.div>
                )}
                {isProcessing && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-blue-400 font-medium"
                  >
                    Processing your query...
                  </motion.div>
                )}
                {isSpeaking && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-green-400 font-medium"
                  >
                    Speaking...
                  </motion.div>
                )}
                {!isListening && !isProcessing && !isSpeaking && (
                  <div className="text-slate-400">
                    Click the microphone to start talking with {activePersona.name}
                  </div>
                )}
              </div>

              {/* Live Transcript */}
              {transcript && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-3 bg-slate-700/50 rounded border border-slate-600/50"
                >
                  <div className="text-sm text-slate-400 mb-1">You said:</div>
                  <div className="text-white">{transcript}</div>
                </motion.div>
              )}
            </div>

            {/* Conversation History */}
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-600/50 rounded-lg p-6">
              <h2 className="text-lg font-semibold text-white mb-4">
                Conversation with {activePersona.name}
              </h2>

              <div className="space-y-4 max-h-96 overflow-y-auto">
                <AnimatePresence>
                  {conversation.map((message) => (
                    <motion.div
                      key={message.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      className={`flex gap-3 ${
                        message.role === 'user' ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      {message.role === 'assistant' && (
                        <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center flex-shrink-0">
                          <Bot className="w-4 h-4 text-white" />
                        </div>
                      )}
                      
                      <div
                        className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg ${
                          message.role === 'user'
                            ? 'bg-blue-500 text-white'
                            : 'bg-slate-700/50 text-slate-200'
                        }`}
                      >
                        <div className="text-sm">{message.content}</div>
                        <div className="text-xs opacity-70 mt-1">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </div>
                        
                        {message.audioUrl && message.role === 'assistant' && (
                          <button
                            onClick={() => {
                              // Play audio response
                              if (audioRef.current) {
                                audioRef.current.src = message.audioUrl!
                                audioRef.current.play()
                              }
                            }}
                            className="mt-2 p-1 bg-slate-600 hover:bg-slate-500 rounded transition-colors"
                          >
                            <Play className="w-3 h-3 text-white" />
                          </button>
                        )}
                      </div>
                      
                      {message.role === 'user' && (
                        <div className="w-8 h-8 bg-slate-600 rounded-full flex items-center justify-center flex-shrink-0">
                          <User className="w-4 h-4 text-white" />
                        </div>
                      )}
                    </motion.div>
                  ))}
                </AnimatePresence>

                {conversation.length === 0 && (
                  <div className="text-center py-8">
                    <Bot className="w-12 h-12 text-slate-500 mx-auto mb-3" />
                    <p className="text-slate-400">
                      Start a conversation by clicking the microphone and asking about satellite imagery
                    </p>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {!activePersona && (
          <div className="text-center py-12">
            <Bot className="w-16 h-16 text-slate-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">Select an AI Persona</h3>
            <p className="text-slate-400">
              Choose a specialized AI assistant to help with your Earth observation analysis
            </p>
          </div>
        )}

        {/* Hidden audio element for TTS playback */}
        <audio ref={audioRef} className="hidden" />
      </div>
    </div>
  )
}

export default VoiceInterface
