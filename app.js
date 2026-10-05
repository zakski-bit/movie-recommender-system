/**
 * CineMatch — MovieLens Recommendation Engine
 * Philosophy: Minimal code, high efficiency, zero AI-slop, secure DOM escaping.
 */

// State
let moviesData = [];
let cfUsersData = {};
let selectedMovie = null;
let currentTab = 'content-based';

// DOM Elements
const statusBadge = document.getElementById('statusBadge');
const movieSearchInput = document.getElementById('movieSearchInput');
const clearSearchBtn = document.getElementById('clearSearchBtn');
const autocompleteList = document.getElementById('autocompleteList');
const activeMovieCard = document.getElementById('activeMovieCard');
const targetTitle = document.getElementById('targetTitle');
const targetYear = document.getElementById('targetYear');
const targetGenres = document.getElementById('targetGenres');
const targetRating = document.getElementById('targetRating');
const targetVotes = document.getElementById('targetVotes');
const resultsHeader = document.getElementById('resultsHeader');
const recommendationsGrid = document.getElementById('recommendationsGrid');
const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');
const quickPickGroup = document.getElementById('quickPickGroup');

// CF Elements
const userSelect = document.getElementById('userSelect');
const userDescription = document.getElementById('userDescription');
const userTopRatedList = document.getElementById('userTopRatedList');
const userPredictedList = document.getElementById('userPredictedList');

// Security: Escape HTML strings to prevent XSS
function escapeHTML(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// -------------------------------------------------------------
// 1. DATA INITIALIZATION
// -------------------------------------------------------------
async function initApp() {
  try {
    statusBadge.textContent = 'Memuat Data...';
    
    // Fetch both datasets concurrently
    const [moviesRes, cfRes] = await Promise.all([
      fetch('data/movies.json'),
      fetch('data/cf_users.json')
    ]);

    if (!moviesRes.ok || !cfRes.ok) {
      throw new Error('Gagal memuat berkas data katalog atau profil rekomendasi.');
    }

    moviesData = await moviesRes.json();
    cfUsersData = await cfRes.json();

    statusBadge.textContent = `● ${moviesData.length.toLocaleString('id-ID')} Film Siap`;
    statusBadge.classList.add('status-ready');

    setupEventListeners();
    setupCFSection();

    // Default target movie: "Toy Story (1995)"
    selectMovieByTitle('Toy Story (1995)');

  } catch (err) {
    console.error('Inisialisasi aplikasi gagal:', err);
    statusBadge.textContent = 'Gagal Memuat Data';
    statusBadge.style.color = '#ef4444';
  }
}

// -------------------------------------------------------------
// 2. TAB SWITCHING
// -------------------------------------------------------------
function setupEventListeners() {
  // Tabs
  const tabButtons = document.querySelectorAll('.tab-btn');
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      switchTab(tabId);
    });
  });

  // Search Input with Debounce
  let debounceTimeout = null;
  movieSearchInput.addEventListener('input', (e) => {
    const val = e.target.value.trim();
    clearSearchBtn.style.display = val ? 'block' : 'none';

    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(() => {
      handleAutocomplete(val);
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
    if (!e.target.closest('.search-input-wrapper')) {
      autocompleteList.style.display = 'none';
    }
  });

  // Quick picks
  quickPickGroup.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (chip) {
      const title = chip.getAttribute('data-title');
      selectMovieByTitle(title);
    }
  });

  // Keyboard navigation on search
  movieSearchInput.addEventListener('keydown', handleKeyNavigation);
}

function switchTab(tabId) {
  currentTab = tabId;
  document.querySelectorAll('.tab-btn').forEach(btn => {
    const active = btn.getAttribute('data-tab') === tabId;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', active);
  });

  document.querySelectorAll('.tab-content').forEach(section => {
    section.classList.toggle('active', section.id === `tab-${tabId}`);
  });
}

// -------------------------------------------------------------
// 3. AUTOCOMPLETE & SELECTION
// -------------------------------------------------------------
function handleAutocomplete(query) {
  if (!query || query.length < 2) {
    autocompleteList.style.display = 'none';
    return;
  }

  const q = query.toLowerCase();
  const matches = moviesData
    .filter(m => m.title.toLowerCase().includes(q))
    .slice(0, 8);

  if (matches.length === 0) {
    autocompleteList.innerHTML = `<div class="autocomplete-item"><span class="item-title">Tidak ditemukan film untuk "${escapeHTML(query)}"</span></div>`;
    autocompleteList.style.display = 'block';
    return;
  }

  autocompleteList.innerHTML = matches.map(m => `
    <div class="autocomplete-item" data-id="${m.id}">
      <span class="item-title">${escapeHTML(m.title)}</span>
      <span class="item-genres">${escapeHTML(m.genres.join(' • '))}</span>
    </div>
  `).join('');

  autocompleteList.style.display = 'block';

  // Add click handlers
  autocompleteList.querySelectorAll('.autocomplete-item').forEach(item => {
    item.addEventListener('click', () => {
      const id = parseInt(item.getAttribute('data-id'), 10);
      const movie = moviesData.find(m => m.id === id);
      if (movie) {
        selectMovie(movie);
      }
      autocompleteList.style.display = 'none';
    });
  });
}

let activeIndex = -1;
function handleKeyNavigation(e) {
  const items = autocompleteList.querySelectorAll('.autocomplete-item');
  if (!items.length || autocompleteList.style.display === 'none') return;

  if (e.key === 'ArrowDown') {
    e.preventDefault();
    activeIndex = (activeIndex + 1) % items.length;
    updateActiveItem(items);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    activeIndex = (activeIndex - 1 + items.length) % items.length;
    updateActiveItem(items);
  } else if (e.key === 'Enter') {
    e.preventDefault();
    if (activeIndex >= 0 && activeIndex < items.length) {
      items[activeIndex].click();
    }
  } else if (e.key === 'Escape') {
    autocompleteList.style.display = 'none';
  }
}

function updateActiveItem(items) {
  items.forEach((item, idx) => {
    item.classList.toggle('selected', idx === activeIndex);
  });
}

function selectMovieByTitle(title) {
  const movie = moviesData.find(m => m.title.toLowerCase() === title.toLowerCase()) 
             || moviesData.find(m => m.title.toLowerCase().includes(title.toLowerCase()));
  if (movie) {
    selectMovie(movie);
  }
}

function selectMovie(movie) {
  selectedMovie = movie;
  movieSearchInput.value = movie.title;
  clearSearchBtn.style.display = 'block';
  autocompleteList.style.display = 'none';

  // Render Target Movie Details
  targetTitle.textContent = movie.title;
  targetYear.textContent = movie.year ? `Tahun: ${movie.year}` : '';
  targetRating.textContent = movie.rating > 0 ? `★ ${movie.rating} / 5.0` : 'Belum ada rating';
  targetVotes.textContent = movie.votes > 0 ? `(${movie.votes.toLocaleString('id-ID')} ulasan)` : '';

  targetGenres.innerHTML = movie.genres.map(g => `
    <span class="genre-pill highlight">${escapeHTML(g)}</span>
  `).join('');

  activeMovieCard.style.display = 'block';

  // Calculate & Render Recommendations
  runContentBasedRecommendations(movie);
}

// -------------------------------------------------------------
// 4. CONTENT-BASED SIMILARITY ENGINE (IN-BROWSER COSINE SIMILARITY)
// -------------------------------------------------------------
function runContentBasedRecommendations(target) {
  loadingState.style.display = 'block';
  recommendationsGrid.innerHTML = '';
  emptyState.style.display = 'none';
  resultsHeader.style.display = 'none';

  // Calculate TF-IDF Cosine Similarity against all 9,737 movies
  const targetVec = target.tfidf || {};
  const targetGenresSet = new Set(target.genres);

  const scoredMovies = [];

  for (let i = 0; i < moviesData.length; i++) {
    const candidate = moviesData[i];
    if (candidate.id === target.id) continue; // Skip target itself

    const candidateVec = candidate.tfidf || {};
    let dotProduct = 0;

    // Dot product on sparse tfidf dictionary
    for (const token in targetVec) {
      if (candidateVec[token]) {
        dotProduct += targetVec[token] * candidateVec[token];
      }
    }

    if (dotProduct > 0.001) {
      // Find matching genres
      const matchGenres = candidate.genres.filter(g => targetGenresSet.has(g));
      scoredMovies.push({
        movie: candidate,
        score: Math.min(1.0, dotProduct),
        matchCount: matchGenres.length,
        matchGenres: matchGenres
      });
    }
  }

  // Sort by score descending, then by vote count
  scoredMovies.sort((a, b) => {
    if (Math.abs(b.score - a.score) > 0.0001) {
      return b.score - a.score;
    }
    return b.movie.votes - a.movie.votes;
  });

  const top10 = scoredMovies.slice(0, 10);

  loadingState.style.display = 'none';
  resultsHeader.style.display = 'flex';

  if (top10.length === 0) {
    recommendationsGrid.innerHTML = `
      <div class="empty-state" style="grid-column: 1 / -1;">
        <p>Tidak ditemukan film dengan kemiripan genre yang cukup tinggi untuk film ini.</p>
      </div>
    `;
    return;
  }

  // Calculate precision@10 (relevant if matchCount >= 2 or candidate genres overlap)
  const relevantCount = top10.filter(item => item.matchCount >= 2 || item.score >= 0.5).length;
  const precisionVal = Math.round((relevantCount / top10.length) * 100);
  document.getElementById('cbPrecision').textContent = `${precisionVal}%`;

  // Render cards
  recommendationsGrid.innerHTML = top10.map((item, idx) => {
    const m = item.movie;
    const matchPct = Math.round(item.score * 100);
    return `
      <div class="movie-card">
        <div class="card-top">
          <span class="card-rank">#${idx + 1}</span>
          <span class="similarity-badge">${matchPct}% Match</span>
        </div>
        <div class="similarity-bar-bg">
          <div class="similarity-bar-fill" style="width: ${matchPct}%;"></div>
        </div>
        <h4 class="card-title">${escapeHTML(m.title)}</h4>
        <div class="card-genres">
          ${m.genres.map(g => {
            const isMatch = targetGenresSet.has(g);
            return `<span class="genre-pill ${isMatch ? 'highlight' : ''}">${escapeHTML(g)}</span>`;
          }).join('')}
        </div>
        <div class="card-actions">
          <span class="text-muted">Cosine: ${item.score.toFixed(3)}</span>
          <button class="btn-select-target" onclick="selectMovieByTitle('${escapeHTML(m.title.replace(/'/g, "\\'"))}')">
            Jadikan Acuan →
          </button>
        </div>
      </div>
    `;
  }).join('');
}

// -------------------------------------------------------------
// 5. COLLABORATIVE FILTERING TAB
// -------------------------------------------------------------
function setupCFSection() {
  const userIds = Object.keys(cfUsersData);
  if (!userIds.length) return;

  userSelect.innerHTML = userIds.map(uid => `
    <option value="${uid}">${escapeHTML(cfUsersData[uid].label)}</option>
  `).join('');

  userSelect.addEventListener('change', () => {
    const uid = userSelect.value;
    renderUserProfile(uid);
  });

  // Default to User 133
  userSelect.value = '133';
  renderUserProfile('133');
}

function renderUserProfile(userId) {
  const profile = cfUsersData[userId];
  if (!profile) return;

  userDescription.textContent = `Jumlah riwayat penilaian pengguna ini: ${profile.ratings_count.toLocaleString('id-ID')} film.`;

  // Render Top Rated Past Movies
  userTopRatedList.innerHTML = profile.top_rated.map((item, idx) => `
    <div class="cf-item">
      <div class="cf-item-left">
        <span class="cf-rank-num">★</span>
        <div class="cf-item-info">
          <span class="cf-item-title">${escapeHTML(item.title)}</span>
          <span class="cf-item-genres">${escapeHTML(item.genres.join(' • '))}</span>
        </div>
      </div>
      <span class="cf-score-pill">${item.rating.toFixed(1)}</span>
    </div>
  `).join('');

  // Render Predicted Recommendations
  userPredictedList.innerHTML = profile.recommendations.map(item => `
    <div class="cf-item">
      <div class="cf-item-left">
        <span class="cf-rank-num">#${item.rank}</span>
        <div class="cf-item-info">
          <span class="cf-item-title">${escapeHTML(item.title)}</span>
          <span class="cf-item-genres">${escapeHTML(item.genres.join(' • '))}</span>
        </div>
      </div>
      <span class="cf-score-pill top-score">Prediksi: ${item.predicted_rating.toFixed(2)}</span>
    </div>
  `).join('');
}

// Global scope binding for inline onclick
window.selectMovieByTitle = selectMovieByTitle;

// Boot
document.addEventListener('DOMContentLoaded', initApp);
