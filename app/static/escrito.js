document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const chatTitle = document.getElementById('chat-title');
    const chatWelcomeMessageHeader = document.getElementById('chat-welcome-message-header'); 
    const langButtons = document.querySelectorAll('.lang-button');
    // REMOVED: const typingIndicatorContainer = document.getElementById('typing-indicator-container');

    // Language-specific texts
    const uiTexts = {
        en: {
            title: "🏦 Banking Assistant",
            welcomeHeader: "Hello! How can I help you today?",
            initialMessage: "Hello! I'm your friendly Banking Assistant. How can I help you today?",
            placeholder: "Type your message...",
            error_server: "Sorry, I encountered an error communicating with the server. Please try again later.",
            error_network: "An error occurred while connecting to the server. Please check your network connection."
            // REMOVED: placeholder_disabled as it's not needed with inline indicator
        },
        es: {
            title: "🏦 Asistente Bancario",
            welcomeHeader: "¡Hola! ¿Cómo puedo ayudarte hoy?",
            initialMessage: "¡Hola! Soy tu Asistente Bancario. ¿Cómo puedo ayudarte hoy?",
            placeholder: "Escribe tu mensaje...",
            error_server: "Lo siento, encontré un error al comunicarme con el servidor. Por favor, inténtalo de nuevo más tarde.",
            error_network: "Ocurrió un error al conectar con el servidor. Por favor, verifica tu conexión a la red."
            // REMOVED: placeholder_disabled
        }
    };

    let currentLanguage = localStorage.getItem('chatLanguage') || 'en';
    let inlineTypingIndicator = null; // To store the reference to the inline indicator

    // Function to update UI elements based on selected language
    function updateUIForLanguage(lang) {
        currentLanguage = lang;
        localStorage.setItem('chatLanguage', lang);

        chatTitle.textContent = uiTexts[lang].title;
        chatWelcomeMessageHeader.textContent = uiTexts[lang].welcomeHeader;
        userInput.placeholder = uiTexts[lang].placeholder;

        langButtons.forEach(button => {
            if (button.dataset.lang === lang) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });
    }

    // Initialize UI with preferred language on load
    updateUIForLanguage(currentLanguage);

    // Add the initial assistant message ONLY if the chat is empty on page load.
    if (chatMessages.children.length === 0) {
        appendMessage('assistant', uiTexts[currentLanguage].initialMessage);
    }

    // Add event listeners for language toggle buttons
    langButtons.forEach(button => {
        button.addEventListener('click', () => {
            const lang = button.dataset.lang;
            if (lang !== currentLanguage) {
                updateUIForLanguage(lang);
            }
        });
    });

    // Function to append messages to the chat interface
    function appendMessage(role, text, sql_query = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${role}-message`);
        messageDiv.classList.add('animated-message');

        let formattedText = text;
        
        if (formattedText.includes('\n* ')) {
            const lines = formattedText.split('\n');
            let inList = false;
            let htmlContent = '';
            
            lines.forEach(line => {
                if (line.startsWith('* ')) {
                    if (!inList) {
                        htmlContent += '<ul>';
                        inList = true;
                    }
                    let listItemContent = line.substring(2).replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                    htmlContent += `<li>${listItemContent}</li>`;
                } else {
                    if (inList) {
                        htmlContent += '</ul>';
                        inList = false;
                    }
                    let paragraphContent = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                    htmlContent += `<p>${paragraphContent}</p>`;
                }
            });
            if (inList) {
                htmlContent += '</ul>';
            }
            formattedText = htmlContent;
        } else {
            formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            formattedText = `<p>${formattedText}</p>`;
        }

        const p = document.createElement('div');
        p.innerHTML = formattedText;
        messageDiv.appendChild(p);

        if (sql_query) {
            const sqlCode = document.createElement('pre');
            sqlCode.classList.add('sql-code');
            sqlCode.textContent = `SQL Query:\n${sql_query}`;
            messageDiv.appendChild(sqlCode);
        }

        chatMessages.appendChild(messageDiv);

        // Ensure animation plays by reflowing
        void messageDiv.offsetWidth; 
        messageDiv.style.opacity = 1;
        messageDiv.style.transform = 'translateY(0)';

        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // NEW: Function to show inline typing indicator
    function showInlineTypingIndicator() {
        if (!inlineTypingIndicator) {
            inlineTypingIndicator = document.createElement('div');
            inlineTypingIndicator.classList.add('inline-typing-indicator');
            for (let i = 0; i < 3; i++) {
                const dot = document.createElement('span');
                inlineTypingIndicator.appendChild(dot);
            }
            chatMessages.appendChild(inlineTypingIndicator);
        }
        chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to bottom
    }

    // NEW: Function to hide inline typing indicator
    function hideInlineTypingIndicator() {
        if (inlineTypingIndicator && inlineTypingIndicator.parentNode) {
            inlineTypingIndicator.parentNode.removeChild(inlineTypingIndicator);
            inlineTypingIndicator = null;
        }
    }

    // Function to send user query to Flask backend
    async function sendMessage() {
        const query = userInput.value.trim();
        if (query === '') return;

        appendMessage('user', query); // Append user's message
        userInput.value = '';
        sendButton.disabled = true;
        userInput.disabled = true;

        showInlineTypingIndicator(); // Show the new inline indicator

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    language: currentLanguage
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error('Backend error:', errorData);
                appendMessage('assistant', uiTexts[currentLanguage].error_server);
                return;
            }

            const data = await response.json();
            appendMessage('assistant', data.response, data.sql_query);

        } catch (error) {
            console.error('Error sending message:', error);
            appendMessage('assistant', uiTexts[currentLanguage].error_network);
        } finally {
            hideInlineTypingIndicator(); // Hide the inline indicator
            sendButton.disabled = false;
            userInput.disabled = false;
            userInput.focus();
        }
    }

    // Event listeners for sending message
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});