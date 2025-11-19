// Общий JavaScript для всех страниц

// Мобильное меню
document.addEventListener('DOMContentLoaded', function() {
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');
    
    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', function() {
            navLinks.classList.toggle('active');
        });
    }

    // Плавная прокрутка для якорных ссылок
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });
                
                // Закрытие мобильного меню после клика
                if (navLinks) {
                    navLinks.classList.remove('active');
                }
            }
        });
    });
});

// Система данных
class MMADataSystem {
    constructor() {
        this.news = [
            {
                title: "Islam Makhachev Shows Intense Wrestling Training Ahead of Title Defense",
                content: "The lightweight champion posted footage of grueling wrestling sessions with Olympic-level partners, hinting at summer return.",
                date: new Date().toLocaleDateString(),
                source: "Instagram @islam_makhachev"
            },
            {
                title: "Sean O'Malley Confirms Bicep Fully Healed, Targets Merab Dvalishvili",
                content: "The bantamweight champion revealed his injury recovery is complete during UFC 303 press conference appearance.",
                date: new Date().toLocaleDateString(),
                source: "UFC Press Conference"
            },
            {
                title: "Khamzat Chimaev Back in Training After Respiratory Illness",
                content: "'Borz' shares training footage showing return to full intensity following health scare that canceled Whittaker fight.",
                date: new Date().toLocaleDateString(),
                source: "Twitter @KChimaev"
            },
            {
                title: "Alex Pereira Spotted Training With Glover Teixeira for Prochazka Rematch",
                content: "The light heavyweight champion working extensively on ground game with former champion ahead of UFC 303 main event.",
                date: new Date().toLocaleDateString(),
                source: "Teixeira MMA Instagram"
            },
            {
                title: "UFC Announces Expansion to New International Markets in 2025",
                content: "Dana White reveals plans for events in Africa and return to Australia with stadium shows scheduled.",
                date: new Date().toLocaleDateString(),
                source: "UFC Official"
            },
            {
                title: "Jon Jones vs Stipe Miocic Targeted for November MSG Event",
                content: "Heavyweight champion recovery on track for legacy fight at Madison Square Garden, per manager announcement.",
                date: new Date().toLocaleDateString(),
                source: "MMA Fighting"
            }
        ];

        this.fights = [
            {
                event: "UFC 303",
                date: "2024-06-29",
                mainEvent: true,
                fighterA: "Alex Pereira",
                fighterB: "Jiri Prochazka",
                oddsA: "-150",
                oddsB: "+130",
                weightClass: "Light Heavyweight"
            },
            {
                event: "UFC 303",
                date: "2024-06-29",
                mainEvent: false,
                fighterA: "Brian Ortega",
                fighterB: "Diego Lopes",
                oddsA: "+110",
                oddsB: "-130",
                weightClass: "Featherweight"
            },
            {
                event: "UFC 303",
                date: "2024-06-29",
                mainEvent: false,
                fighterA: "Anthony Smith",
                fighterB: "Roman Dolidze",
                oddsA: "+150",
                oddsB: "-180",
                weightClass: "Light Heavyweight"
            },
            {
                event: "UFC Fight Night: Whittaker vs Aliskerov",
                date: "2024-06-22",
                mainEvent: true,
                fighterA: "Robert Whittaker",
                fighterB: "Ikram Aliskerov",
                oddsA: "-280",
                oddsB: "+230",
                weightClass: "Middleweight"
            }
        ];
    }

    getNews() {
        return new Promise(resolve => {
            setTimeout(() => resolve(this.news), 500);
        });
    }

    getFights() {
        return new Promise(resolve => {
            setTimeout(() => resolve(this.fights), 500);
        });
    }
}

// Функции для главной страницы
function loadNews() {
    const newsContainer = document.getElementById('news-container');
    if (!newsContainer) return;

    newsContainer.classList.add('loading');
    
    const dataSystem = new MMADataSystem();
    dataSystem.getNews().then(news => {
        newsContainer.innerHTML = news.map(item => `
            <div class="news-card">
                <div class="news-date">${item.date} • ${item.source}</div>
                <h3>${item.title}</h3>
                <p>${item.content}</p>
            </div>
        `).join('');
        
        newsContainer.classList.remove('loading');
    });
}

function loadFights() {
    const fightsContainer = document.getElementById('fights-container');
    if (!fightsContainer) return;

    fightsContainer.classList.add('loading');
    
    const dataSystem = new MMADataSystem();
    dataSystem.getFights().then(fights => {
        const fightsHTML = `
            <table class="fights-table">
                <thead>
                    <tr>
                        <th>Fighter A</th>
                        <th>Fighter B</th>
                        <th>Odds (A/B)</th>
                        <th>Weight Class</th>
                        <th>Event</th>
                    </tr>
                </thead>
                <tbody>
                    ${fights.map(fight => `
                        <tr>
                            <td>${fight.fighterA} ${fight.mainEvent ? '🏆' : ''}</td>
                            <td>${fight.fighterB}</td>
                            <td>
                                <span class="odds ${parseFloat(fight.oddsA) < 0 ? 'positive' : 'negative'}">${fight.oddsA}</span> / 
                                <span class="odds ${parseFloat(fight.oddsB) < 0 ? 'positive' : 'negative'}">${fight.oddsB}</span>
                            </td>
                            <td>${fight.weightClass}</td>
                            <td>${fight.event}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        
        fightsContainer.innerHTML = fightsHTML;
        fightsContainer.classList.remove('loading');
    });
}

function startCountdown() {
    function updateCountdown() {
        const eventDate = new Date('June 29, 2024 22:00:00').getTime();
        const now = new Date().getTime();
        const timeLeft = eventDate - now;

        const days = Math.floor(timeLeft / (1000 * 60 * 60 * 24));
        const hours = Math.floor((timeLeft % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((timeLeft % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((timeLeft % (1000 * 60)) / 1000);

        const daysEl = document.getElementById('days');
        const hoursEl = document.getElementById('hours');
        const minutesEl = document.getElementById('minutes');
        const secondsEl = document.getElementById('seconds');

        if (daysEl) daysEl.textContent = days.toString().padStart(2, '0');
        if (hoursEl) hoursEl.textContent = hours.toString().padStart(2, '0');
        if (minutesEl) minutesEl.textContent = minutes.toString().padStart(2, '0');
        if (secondsEl) secondsEl.textContent = seconds.toString().padStart(2, '0');
    }

    updateCountdown();
    setInterval(updateCountdown, 1000);
}