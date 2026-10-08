/**
 * CineMatch — MovieLens Recommendation Engine
 * Streaming Interface & Real-time Client ML Engine
 * Philosophy: Pure typographic & SVG vector design, ZERO emojis, secure DOM escaping, zero-latency.
 */

// Global State
let moviesData = [];
let cfUsersData = {};
let featuredMovies = [];
let selectedMovie = null;

// DOM Elements - Navigation & Catalog
const catalogStatus = document.getElementById('catalogStatus');
const mosaicGrid = document.getElementById('mosaicGrid');
const popularCarousel = document.getElementById('popularCarousel');

// DOM Elements - Content-Based
const movieSearchInput = document.getElementById('movieSearchInput');
const clearSearchBtn = document.getElementById('clearSearchBtn');
const autocompleteList = document.getElementById('autocompleteList');
const quickPickGroup = document.getElementById('quickPickGroup');
const activeMovieSpotlight = document.getElementById('activeMovieSpotlight');
const spotlightPoster = document.getElementById('spotlightPoster');
const spotlightTitle = document.getElementById('spotlightTitle');
const spotlightYear = document.getElementById('spotlightYear');
const spotlightRating = document.getElementById('spotlightRating');
const spotlightVotes = document.getElementById('spotlightVotes');
const spotlightGenres = document.getElementById('spotlightGenres');
const cbResultsBar = document.getElementById('cbResultsBar');
const cbPrecision = document.getElementById('cbPrecision');
const cbRecommendationsGrid = document.getElementById('cbRecommendationsGrid');
const cbEmptyState = document.getElementById('cbEmptyState');
const cbLoadingState = document.getElementById('cbLoadingState');

// DOM Elements - Collaborative
const userSelectDropdown = document.getElementById('userSelectDropdown');
const userPickerDetails = document.getElementById('userPickerDetails');
const cfFavoritesList = document.getElementById('cfFavoritesList');
const cfPredictedList = document.getElementById('cfPredictedList');

// ==========================================================================
// 1. SECURITY & HELPER FUNCTIONS
// ==========================================================================
function escapeHTML(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function getPosterUrl(movie) {
  if (movie && movie.poster && movie.poster.startsWith('http')) {
    return movie.poster;
  }
  // Generate clean dark slate SVG placeholder with minimalist geometric vector symbol (NO EMOJIS)
  const title = movie && movie.title ? movie.title : 'Film';
  const cleanTitle = title.length > 25 ? title.slice(0, 23) + '...' : title;
  const genres = movie && movie.genres ? movie.genres.slice(0, 2).join(' • ') : 'MovieLens';

  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(`
    <svg xmlns="http://www.w3.org/2000/svg" width="300" height="450" viewBox="0 0 300 450">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#191c24"/>
          <stop offset="100%" stop-color="#0c0e12"/>
        </linearGradient>
      </defs>
      <rect width="300" height="450" fill="url(#g)"/>
      <rect x="12" y="12" width="276" height="426" rx="8" fill="none" stroke="#252a36" stroke-width="2"/>
      <g transform="translate(132, 160)" stroke="#f59e0b" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
        <rect x="0" y="0" width="36" height="26" rx="4"/>
        <path d="m0 8 36 0"/>
        <path d="m8 0 4 8"/>
        <path d="m18 0 4 8"/>
        <path d="m28 0 4 8"/>
      </g>
      <text x="150" y="235" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="bold" font-size="15" fill="#f8fafc" text-anchor="middle">
        ${cleanTitle}
      </text>
      <text x="150" y="265" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11" fill="#64748b" text-anchor="middle">
        ${genres}
      </text>
    </svg>
  `)}`;
}

// Vector Star SVG Helper
function getStarSvg(size = 12) {
  return `<svg class="vector-star" width="${size}" height="${size}" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>`;
}

// ==========================================================================
// 2. INITIALIZATION
// ==========================================================================
async function initApp() {
  try {
    catalogStatus.innerHTML = '<span class="status-dot"></span><span>Memuat Katalog...</span>';

    // Load all data concurrently
    const [moviesRes, cfRes, featRes] = await Promise.all([
      fetch('data/movies.json'),
      fetch('data/cf_users.json'),
      fetch('data/featured_movies.json')
    ]);

    if (!moviesRes.ok || !cfRes.ok) {
      throw new Error('Gagal memuat berkas data katalog atau profil.');
    }

    moviesData = await moviesRes.json();
    cfUsersData = await cfRes.json();
    featuredMovies = featRes.ok ? await featRes.json() : moviesData.slice(0, 24);

    catalogStatus.innerHTML = `<span class="status-dot"></span><span>${moviesData.length.toLocaleString('id-ID')} Film Siap</span>`;

    // Render Components
    renderHeroMosaic();
    renderPopularCarousel();
    setupSearchAndAutocomplete();
    setupCollaborativeSection();

    // Default active movie: "Toy Story (1995)"
    selectMovieByTitle('Toy Story (1995)', false);

  } catch (err) {
    console.error('Inisialisasi aplikasi gagal:', err);
    catalogStatus.innerHTML = '<span class="status-dot status-error"></span><span>Gagal Memuat Data</span>';
    catalogStatus.style.color = '#ef4444';
  }
}

// ==========================================================================
// 3. HERO MOSAIC POSTER WALL (LIKE PRIME VIDEO)
// ==========================================================================
function renderHeroMosaic() {
  if (!mosaicGrid) return;

  // Take 9 top iconic movies with posters for 3x3 mosaic
  const mosaicMovies = featuredMovies.slice(0, 9);

  mosaicGrid.innerHTML = mosaicMovies.map(m => `
    <div class="mosaic-item" onclick="selectMovieByTitle('${escapeHTML(m.title.replace(/'/g, "\\'"))}', true)" title="Analisis Rekomendasi: ${escapeHTML(m.title)}">
      <img class="mosaic-img" src="${getPosterUrl(m)}" alt="${escapeHTML(m.title)}" loading="lazy">
      <div class="mosaic-overlay">
        <span class="mosaic-title">${escapeHTML(m.title)}</span>
        <span class="mosaic-year">${m.year || ''}</span>
      </div>
    </div>
  `).join('');
}

// ==========================================================================
// 4. STREAMING CAROUSEL (POPULAR MOVIES)
// ==========================================================================
function renderPopularCarousel() {
  if (!popularCarousel) return;

  popularCarousel.innerHTML = featuredMovies.map(m => `
    <div class="carousel-card" onclick="selectMovieByTitle('${escapeHTML(m.title.replace(/'/g, "\\'"))}', true)">
      <div class="carousel-poster-wrap">
        <img class="carousel-poster-img" src="${getPosterUrl(m)}" alt="${escapeHTML(m.title)}" loading="lazy">
        <span class="carousel-badge-top">
          ${getStarSvg(10)}
          <span>${m.rating || 4.0}</span>
        </span>
      </div>
      <div class="carousel-info">
        <h4 class="carousel-title" title="${escapeHTML(m.title)}">${escapeHTML(m.title)}</h4>
        <div class="carousel-meta-row">
          <span>${m.year || ''}</span>
          <span>${m.votes ? m.votes + ' ulasan' : ''}</span>
        </div>
        <button class="carousel-cta">Rekomendasikan Serupa</button>
      </div>
    </div>
  `).join('');
}

// ==========================================================================
// 5. SEARCH & AUTOCOMPLETE
// ==========================================================================
function setupSearchAndAutocomplete() {
  let debounceTimer = null;

  movieSearchInput.addEventListener('input', (e) => {
    const val = e.target.value.trim();
    clearSearchBtn.style.display = val ? 'block' : 'none';

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      renderAutocomplete(val);
    }, 150);
  });

  clearSearchBtn.addEventListener('click', () => {
    movieSearchInput.value = '';
    clearSearchBtn.style.display = 'none';
    autocompleteList.style.display = 'none';
    movieSearchInput.focus();
  });

  // Close autocomplete on click outside
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-field-wrapper')) {
      autocompleteList.style.display = 'none';
    }
  });

  // Quick picks
  quickPickGroup.addEventListener('click', (e) => {
    const chip = e.target.closest('.quick-chip');
    if (chip) {
      const title = chip.getAttribute('data-title');
      selectMovieByTitle(title, true);
    }
  });

  // Keyboard navigation
  movieSearchInput.addEventListener('keydown', handleSearchKeyNav);
}

function renderAutocomplete(query) {
  if (!query || query.length < 2) {
    autocompleteList.style.display = 'none';
    return;
  }

  const q = query.toLowerCase();
  const matches = moviesData
    .filter(m => m.title.toLowerCase().includes(q))
    .slice(0, 7);

  if (matches.length === 0) {
    autocompleteList.innerHTML = `
      <div class="autocomplete-row" style="cursor: default;">
        <div class="auto-info">
          <span class="auto-title">Tidak ditemukan film untuk "${escapeHTML(query)}"</span>
        </div>
      </div>
    `;
    autocompleteList.style.display = 'block';
    return;
  }

  autocompleteList.innerHTML = matches.map(m => `
    <div class="autocomplete-row" data-id="${m.id}">
      <img class="auto-poster-thumb" src="${getPosterUrl(m)}" alt="" loading="lazy">
      <div class="auto-info">
        <span class="auto-title">${escapeHTML(m.title)}</span>
        <span class="auto-genres">${escapeHTML((m.genres || []).join(' • '))}</span>
      </div>
    </div>
  `).join('');

  autocompleteList.style.display = 'block';

  // Attach click events
  autocompleteList.querySelectorAll('.autocomplete-row').forEach(row => {
    row.addEventListener('click', () => {
      const id = parseInt(row.getAttribute('data-id'), 10);
      const movie = moviesData.find(m => m.id === id);
      if (movie) {
        selectMovie(movie, true);
      }
      autocompleteList.style.display = 'none';
    });
  });
}

let activeNavIndex = -1;
function handleSearchKeyNav(e) {
  const items = autocompleteList.querySelectorAll('.autocomplete-row');
  if (!items.length || autocompleteList.style.display === 'none') return;

  if (e.key === 'ArrowDown') {
    e.preventDefault();
    activeNavIndex = (activeNavIndex + 1) % items.length;
    items.forEach((item, idx) => item.classList.toggle('selected', idx === activeNavIndex));
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    activeNavIndex = (activeNavIndex - 1 + items.length) % items.length;
    items.forEach((item, idx) => item.classList.toggle('selected', idx === activeNavIndex));
  } else if (e.key === 'Enter') {
    e.preventDefault();
    if (activeNavIndex >= 0 && activeNavIndex < items.length) {
      items[activeNavIndex].click();
    }
  } else if (e.key === 'Escape') {
    autocompleteList.style.display = 'none';
  }
}

// ==========================================================================
// 6. CONTENT-BASED RECOMMENDATION ENGINE (IN-BROWSER COSINE SIMILARITY)
// ==========================================================================
function selectMovieByTitle(title, shouldScroll = false) {
  const clean = title.toLowerCase().trim();
  const movie = moviesData.find(m => m.title.toLowerCase().trim() === clean)
             || moviesData.find(m => m.title.toLowerCase().includes(clean));

  if (movie) {
    selectMovie(movie, shouldScroll);
  }
}

function selectMovie(movie, shouldScroll = false) {
  selectedMovie = movie;
  movieSearchInput.value = movie.title;
  clearSearchBtn.style.display = 'block';
  autocompleteList.style.display = 'none';

  // Render Spotlight Card
  spotlightPoster.src = getPosterUrl(movie);
  spotlightTitle.textContent = movie.title;
  spotlightYear.textContent = movie.year ? `Tahun: ${movie.year}` : '';
  spotlightRating.innerHTML = movie.rating > 0 
    ? `${getStarSvg(13)} <span>${movie.rating} / 5.0</span>` 
    : '<span>Belum ada rating</span>';
  spotlightVotes.textContent = movie.votes > 0 ? `${movie.votes.toLocaleString('id-ID')} ulasan komunitas` : 'Film katalog';

  spotlightGenres.innerHTML = (movie.genres || []).map(g => `
    <span class="genre-tag highlight">${escapeHTML(g)}</span>
  `).join('');

  activeMovieSpotlight.style.display = 'grid';

  // Run Real-time In-browser Cosine Similarity
  runContentBasedRecommendations(movie);

  // Smooth scroll to recommendation area if requested
  if (shouldScroll) {
    const targetEl = document.getElementById('content-based-section');
    if (targetEl) {
      targetEl.scrollIntoView({ behavior: 'smooth' });
    }
  }
}

function runContentBasedRecommendations(target) {
  cbLoadingState.style.display = 'block';
  cbRecommendationsGrid.innerHTML = '';
  cbEmptyState.style.display = 'none';
  cbResultsBar.style.display = 'none';

  const targetVec = target.tfidf || {};
  const targetGenresSet = new Set(target.genres || []);

  const scoredMovies = [];

  for (let i = 0; i < moviesData.length; i++) {
    const candidate = moviesData[i];
    if (candidate.id === target.id) continue;

    const candidateVec = candidate.tfidf || {};
    let dotProduct = 0;

    for (const token in targetVec) {
      if (candidateVec[token]) {
        dotProduct += targetVec[token] * candidateVec[token];
      }
    }

    if (dotProduct > 0.001) {
      const matchGenres = (candidate.genres || []).filter(g => targetGenresSet.has(g));
      scoredMovies.push({
        movie: candidate,
        score: Math.min(1.0, dotProduct),
        matchCount: matchGenres.length
      });
    }
  }

  // Sort by Cosine Score descending, then by votes
  scoredMovies.sort((a, b) => {
    if (Math.abs(b.score - a.score) > 0.0001) {
      return b.score - a.score;
    }
    return (b.movie.votes || 0) - (a.movie.votes || 0);
  });

  const top10 = scoredMovies.slice(0, 10);

  cbLoadingState.style.display = 'none';
  cbResultsBar.style.display = 'flex';

  if (top10.length === 0) {
    cbRecommendationsGrid.innerHTML = `
      <div class="empty-state-card" style="grid-column: 1 / -1;">
        <p>Tidak ditemukan film dengan kemiripan genre yang memadai untuk film ini.</p>
      </div>
    `;
    return;
  }

  // Calculate Precision@10 (relevant if matchCount >= 2 or score >= 0.5)
  const relevantCount = top10.filter(item => item.matchCount >= 2 || item.score >= 0.5).length;
  const precisionVal = Math.round((relevantCount / top10.length) * 100);
  cbPrecision.textContent = `${precisionVal}.00%`;

  // Render Top-10 Poster Cards
  cbRecommendationsGrid.innerHTML = top10.map((item, idx) => {
    const m = item.movie;
    const matchPct = Math.round(item.score * 100);
    return `
      <div class="rec-card">
        <div class="rec-poster-box">
          <img class="rec-poster-img" src="${getPosterUrl(m)}" alt="${escapeHTML(m.title)}" loading="lazy">
          <span class="rec-rank-badge">#${idx + 1}</span>
          <span class="rec-score-badge">${matchPct}% Match</span>
        </div>
        <div class="rec-info">
          <div class="rec-title-block">
            <h4 class="rec-title" title="${escapeHTML(m.title)}">${escapeHTML(m.title)}</h4>
            <span class="rec-year">${m.year || ''}</span>
          </div>
          <div class="rec-genres-list">
            ${(m.genres || []).map(g => {
              const isMatch = targetGenresSet.has(g);
              return `<span class="genre-tag ${isMatch ? 'highlight' : ''}">${escapeHTML(g)}</span>`;
            }).join('')}
          </div>
          <div class="rec-action-row">
            <span class="rec-cosine-val">Cos: ${item.score.toFixed(3)}</span>
            <button class="btn-pivot-target" onclick="selectMovieByTitle('${escapeHTML(m.title.replace(/'/g, "\\'"))}', true)">
              Jadikan Acuan →
            </button>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

// ==========================================================================
// 7. COLLABORATIVE FILTERING (DEEP LEARNING RECOMMENDERNET)
// ==========================================================================
function setupCollaborativeSection() {
  const userIds = Object.keys(cfUsersData);
  if (!userIds.length) return;

  userSelectDropdown.innerHTML = userIds.map(uid => `
    <option value="${uid}">${escapeHTML(cfUsersData[uid].label)}</option>
  `).join('');

  userSelectDropdown.addEventListener('change', () => {
    renderUserProfile(userSelectDropdown.value);
  });

  // Default to User 133
  userSelectDropdown.value = '133';
  renderUserProfile('133');
}

function renderUserProfile(userId) {
  const profile = cfUsersData[userId];
  if (!profile) return;

  userPickerDetails.textContent = `Riwayat total penilaian pengguna ini: ${profile.ratings_count.toLocaleString('id-ID')} film pada dataset MovieLens.`;

  // Render Past Favorites List
  cfFavoritesList.innerHTML = profile.top_rated.map(item => `
    <div class="cf-movie-row">
      <img class="cf-row-poster" src="${item.poster || getPosterUrl(item)}" alt="${escapeHTML(item.title)}" loading="lazy">
      <div class="cf-row-center">
        <span class="cf-row-title">${escapeHTML(item.title)}</span>
        <span class="cf-row-genres">${escapeHTML((item.genres || []).join(' • '))}</span>
      </div>
      <span class="cf-row-score rating-gold">
        ${getStarSvg(11)}
        <span>${item.rating.toFixed(1)}</span>
      </span>
    </div>
  `).join('');

  // Render Predicted Recommendations List
  cfPredictedList.innerHTML = profile.recommendations.map(item => `
    <div class="cf-movie-row">
      <img class="cf-row-poster" src="${item.poster || getPosterUrl(item)}" alt="${escapeHTML(item.title)}" loading="lazy">
      <div class="cf-row-center">
        <span class="cf-row-rank">Peringkat #${item.rank}</span>
        <span class="cf-row-title">${escapeHTML(item.title)}</span>
        <span class="cf-row-genres">${escapeHTML((item.genres || []).join(' • '))}</span>
      </div>
      <span class="cf-row-score">Prediksi: ${item.predicted_rating.toFixed(2)}</span>
    </div>
  `).join('');
}

// Global scope binding for inline onclick
window.selectMovieByTitle = selectMovieByTitle;

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', initApp);
