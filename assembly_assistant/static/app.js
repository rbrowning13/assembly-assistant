    const state = {
      jobs: [],
      sections: [],
      packages: [],
      transcripts: [],
      settings: {
        has_openai_api_key: false,
        transcription_model: 'whisper-1',
        transcription_enabled: true,
        default_processing: 'Standard',
        storage_root: 'storage'
      },
      currentJob: null,
      currentPackage: null,
      currentPage: 'jobs',
      showArchived: false,
      search: '',
      sectionSearch: '',
      packageListSearch: '',
      packageSearch: '',
      packageDraft: [],
      packageDraftId: null,
      packageDraftMeta: {
        title: '',
        product: '',
        work_area: '',
        station: '',
        status: 'Draft'
      },
      currentSection: null,
      sectionDraftId: null,
      sectionDraft: null,
      sectionMediaOptions: {},
      sectionMediaPicker: {
        open: false,
        replaceIndex: null,
        mode: 'append',
        filter: 'near',
        search: ''
      },
      sectionStepIndexes: {},
      sectionQaFocus: null,
      currentPackageData: null,
      advancedOpenJobs: {},
      sourceSetupOpen: {},
      expandedPackageSections: {},
      scrollPositions: {},
      selectedChoices: {},
      questionDrafts: {},
      questionAttachmentNotes: {},
      questionFiles: {},
      questionFileObjects: {},
      selectedStages: {},
      draftSlideIndexes: {},
      operatorStepIndexes: {},
      renderFingerprint: '',
      lastScrollAt: 0,
      poller: null
    };

    const stageKeys = ['upload_check', 'transcription', 'drawings', 'images', 'video', 'questions', 'draft_sections'];
    const stageInitials = {upload_check:'U', transcription:'T', drawings:'D', images:'I', video:'V', questions:'Q', draft_sections:'S'};

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
    }

    function mediaUrl(path) {
      return `/media?path=${encodeURIComponent(path || '')}`;
    }

    function defaultMediaSetting() {
      return {fit: 'contain', aspect: 'auto', crop_position: 'center'};
    }

    function normalizeMediaSetting(setting = {}) {
      const fit = ['contain', 'cover', 'fill'].includes(setting.fit) ? setting.fit : 'contain';
      const aspect = ['auto', 'portrait', 'landscape', 'square'].includes(setting.aspect) ? setting.aspect : 'auto';
      const cropPosition = ['center', 'top', 'bottom', 'left', 'right'].includes(setting.crop_position || setting.position)
        ? (setting.crop_position || setting.position)
        : 'center';
      return {
        fit: fit === 'fill' ? 'cover' : fit,
        aspect,
        crop_position: cropPosition
      };
    }

    function normalizeStepMediaSettings(step) {
      const media = normalizeMediaList(step?.media || []);
      const raw = step?.media_settings && typeof step.media_settings === 'object' && !Array.isArray(step.media_settings)
        ? step.media_settings
        : {};
      const allowed = new Set(media);
      const settings = {};
      Object.entries(raw).forEach(([path, setting]) => {
        if (!allowed.has(path) || !setting || typeof setting !== 'object') return;
        settings[path] = normalizeMediaSetting(setting);
      });
      return settings;
    }

    function mediaSettingFor(step, path) {
      const settings = normalizeStepMediaSettings(step || {});
      return {...defaultMediaSetting(), ...(settings[String(path || '')] || {})};
    }

    function mediaClassNames(setting) {
      const clean = normalizeMediaSetting(setting || {});
      return `fit-${clean.fit} aspect-${clean.aspect} position-${clean.crop_position}`;
    }

    function openImageZoom(path, label = '') {
      if (!path) return;
      const zoom = document.getElementById('imageZoom');
      const img = document.getElementById('imageZoomImg');
      const title = document.getElementById('imageZoomLabel');
      img.src = mediaUrl(path);
      img.alt = label || path;
      title.textContent = label || path;
      zoom.classList.add('open');
    }

    function closeImageZoom() {
      const zoom = document.getElementById('imageZoom');
      const img = document.getElementById('imageZoomImg');
      zoom.classList.remove('open');
      img.src = '';
    }

    function isBrowserPreviewImage(file) {
      const type = String(file?.type || '').toLowerCase();
      const name = String(file?.name || file?.stored_path || '').toLowerCase();
      return type.startsWith('image/') && /\.(jpe?g|png|webp|gif)$/i.test(name);
    }

    function jsSingleQuoted(value) {
      return String(value ?? '')
        .replace(/\\/g, '\\\\')
        .replace(/'/g, "\\'")
        .replace(/\r/g, '\\r')
        .replace(/\n/g, '\\n');
    }

    function uploadIdentity(file) {
      return file?.sha256 || file?.stored_path || `${file?.name || ''}:${file?.bytes || file?.size || ''}`;
    }

    function compactArtifacts(artifacts) {
      const transcribedAudioSources = new Set(
        (artifacts || [])
          .filter(file => file.kind === 'audio' && file.processing_status === 'transcribed')
          .map(file => file.source_file)
      );
      const seen = new Set();
      return (artifacts || []).filter(file => {
        const key = `${file.kind || ''}:${file.stored_path || file.name || ''}`;
        if (seen.has(key)) return false;
        seen.add(key);
        if (file.kind === 'audio' && String(file.processing_status || '').includes('failed') && transcribedAudioSources.has(file.source_file)) {
          return false;
        }
        return true;
      });
    }

    async function api(path, options = {}) {
      const headers = options.body instanceof FormData ? {} : {'Content-Type': 'application/json'};
      const res = await fetch(path, {
        headers,
        ...options
      });
      if (!res.ok) {
        const text = await res.text();
        try {
          const parsed = JSON.parse(text);
          throw new Error(parsed.error || text);
        } catch (err) {
          if (err instanceof SyntaxError) throw new Error(text);
          throw err;
        }
      }
      return res.json();
    }

    function toast(message) {
      const el = document.getElementById('toast');
      el.textContent = message;
      el.classList.add('show');
      setTimeout(() => el.classList.remove('show'), 2200);
    }

    function setTitle(title) {
      document.getElementById('pageTitle').textContent = title;
      document.querySelectorAll('.nav button').forEach(btn => {
        const activePage = state.currentPage === 'job' ? 'jobs' : state.currentPage === 'section' ? 'sections' : ['package_edit', 'package_view', 'package_print', 'operator_view'].includes(state.currentPage) ? 'package' : state.currentPage;
        btn.classList.toggle('active', btn.dataset.page === activePage);
      });
    }

    function setTopActions(html) {
      document.getElementById('topActions').innerHTML = html || '';
    }

    function invalidateRender() {
      state.renderFingerprint = '';
    }

    function go(page, id = null) {
      invalidateRender();
      state.currentPage = page;
      if (page === 'job' && id) {
        location.hash = `job/${id}`;
      } else if (page === 'section' && id) {
        location.hash = `section/${id}`;
      } else if (page === 'package_view' && id) {
        location.hash = `package/${id}`;
      } else if (page === 'package_edit' && id) {
        location.hash = `package/${id}/edit`;
      } else if (page === 'package_print' && id) {
        location.hash = `package/${id}/print`;
      } else if (page === 'operator_view' && id) {
        location.hash = `package/${id}/operator`;
      } else {
        location.hash = page;
      }
      render();
    }

    function readHash() {
      const pathRoute = location.pathname.replace(/^\/+|\/+$/g, '');
      const hash = location.hash.replace(/^#/, '') || (pathRoute.startsWith('package/') ? pathRoute : pathRoute);
      if (!hash || hash === 'jobs') {
        state.currentPage = 'jobs';
        state.currentJob = null;
        state.currentSection = null;
        state.currentPackage = null;
      } else if (hash.startsWith('job/')) {
        state.currentPage = 'job';
        state.currentJob = hash.split('/')[1];
        state.currentSection = null;
        state.currentPackage = null;
      } else if (hash.startsWith('section/')) {
        state.currentPage = 'section';
        state.currentSection = hash.split('/')[1];
        state.currentJob = null;
        state.currentPackage = null;
      } else if (hash.startsWith('package/')) {
        const parts = hash.split('/');
        state.currentPage = parts[2] === 'print' ? 'package_print' : parts[2] === 'edit' ? 'package_edit' : parts[2] === 'operator' ? 'operator_view' : 'package_view';
        state.currentPackage = parts[1];
        state.currentJob = null;
        state.currentSection = null;
      } else {
        state.currentPage = hash;
        state.currentJob = null;
        state.currentSection = null;
        state.currentPackage = null;
      }
    }

    function toggleNav() {
      document.getElementById('appShell').classList.toggle('nav-closed');
    }

    function isEditingField() {
      const el = document.activeElement;
      if (!el) return false;
      return ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName);
    }

    function isActivelyScrolling() {
      return Date.now() - (state.lastScrollAt || 0) < 900;
    }

    function scrollKey(label) {
      const id = state.currentJob || state.currentSection || state.currentPage;
      return `${state.currentPage}:${id}:${label}`;
    }

    function elementScrollLabel(el, index) {
      if (el.dataset.scrollKey) return el.dataset.scrollKey;
      if (el.id) return `id:${el.id}`;
      if (el.className && typeof el.className === 'string') return `class:${el.className.split(/\s+/).filter(Boolean).join('.')}:${index}`;
      return `node:${el.tagName}:${index}`;
    }

    function scrollableElements() {
      return Array.from(document.querySelectorAll('textarea, .table-wrap, .log, [data-scroll-key], .stage-detail-body, .frame-grid, .media-strip, .panel-body, .details-body')).filter((el, index, arr) => {
        if (arr.indexOf(el) !== index) return false;
        return el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1 || el.dataset.scrollKey;
      });
    }

    function saveScrollPositions() {
      scrollableElements().forEach((el, index) => {
        state.scrollPositions[scrollKey(elementScrollLabel(el, index))] = {
          left: el.scrollLeft,
          top: el.scrollTop
        };
      });
      const page = document.scrollingElement || document.documentElement;
      if (page) state.scrollPositions[scrollKey('page')] = {left: page.scrollLeft, top: page.scrollTop};
    }

    function restoreScrollPositions() {
      const applyRestore = () => {
        autosizeTextareas();
        scrollableElements().forEach((el, index) => {
          const pos = state.scrollPositions[scrollKey(elementScrollLabel(el, index))];
          if (pos) {
            el.scrollLeft = pos.left;
            el.scrollTop = pos.top;
          }
        });
        const page = document.scrollingElement || document.documentElement;
        const pagePos = state.scrollPositions[scrollKey('page')];
        if (page && pagePos) {
          page.scrollLeft = pagePos.left;
          page.scrollTop = pagePos.top;
        }
      };
      requestAnimationFrame(() => {
        applyRestore();
        requestAnimationFrame(applyRestore);
        setTimeout(applyRestore, 80);
      });
    }

    async function loadData() {
      const [jobs, sections, packagesData, settings] = await Promise.all([api('/api/jobs'), api('/api/sections'), api('/api/packages'), api('/api/settings')]);
      state.jobs = jobs.jobs;
      state.sections = sections.sections;
      state.packages = packagesData.packages;
      state.settings = settings.settings;
      if (state.currentJob) {
        const [detail, transcriptData] = await Promise.all([
          api(`/api/jobs/${state.currentJob}`),
          api(`/api/jobs/${state.currentJob}/transcripts`)
        ]);
        state.currentJobData = detail.job;
        state.transcripts = transcriptData.transcripts;
      } else {
        state.transcripts = [];
      }
      if (state.currentPackage) {
        const detail = await api(`/api/packages/${state.currentPackage}`);
        state.currentPackageData = detail.package;
      } else {
        state.currentPackageData = null;
      }
      if (state.currentSection) {
        try {
          const mediaData = await api(`/api/sections/${state.currentSection}/media-options`);
          state.sectionMediaOptions[state.currentSection] = mediaData.media || [];
        } catch (err) {
          state.sectionMediaOptions[state.currentSection] = [];
        }
      }
    }

    function startPolling() {
      if (state.poller) clearInterval(state.poller);
      state.poller = setInterval(async () => {
        if (isEditingField() || isActivelyScrolling()) return;
        saveScrollPositions();
        await loadData();
        if (isEditingField() || isActivelyScrolling()) return;
        if (state.currentPage === 'jobs' || state.currentPage === 'job' || state.currentPage === 'sections' || state.currentPage === 'package' || state.currentPage === 'package_edit' || state.currentPage === 'package_view' || state.currentPage === 'package_print' || state.currentPage === 'operator_view') {
          render(false);
        }
      }, 2500);
    }

    function statusChip(status) {
      let cls = 'chip';
      if (['Draft Sections Ready', 'Complete', 'Ready'].includes(status)) cls += ' good';
      if (['Questions Pending', 'Processing', 'Drafting Sections'].includes(status)) cls += ' warn';
      if (['Failed', 'Cancelled', 'Needs Attention'].includes(status)) cls += ' bad';
      return `<span class="${cls}">${esc(status)}</span>`;
    }

    function dot(stage) {
      return `<span class="dot ${esc(stage.status)}" title="${esc(stage.label)}: ${esc(stage.status)}">${stageInitials[stage.key]}</span>`;
    }

    function stageDots(job) {
      return `<div class="stage-dots">${stageKeys.map(k => dot(job.stages[k])).join('')}</div>`;
    }

    function stageCell(job, key) {
      const s = job.stages[key];
      return `<div class="chip-row">${dot(s)}<span class="subtle">${s.percent}%</span></div>`;
    }

    function autosizeTextareas() {
      document.querySelectorAll('textarea[data-autogrow]').forEach(el => {
        const top = el.scrollTop;
        el.style.height = 'auto';
        const maxHeight = Number(el.dataset.maxHeight || 560);
        el.style.height = `${Math.min(el.scrollHeight + 2, maxHeight)}px`;
        el.scrollTop = top;
      });
    }

    function pageFingerprint() {
      if (state.currentPage === 'job' && state.currentJobData) {
        const job = state.currentJobData;
        return JSON.stringify({
          page: state.currentPage,
          job: job.id,
          status: job.status,
          progress: job.progress,
          selectedStage: state.selectedStages[job.id] || activeStageKey(job),
          advanced: !!state.advancedOpenJobs[job.id],
          stageNotes: stageKeys.map(key => [key, job.stages[key]?.status, job.stages[key]?.percent, job.stages[key]?.note]),
          questions: (job.questions || []).map(q => [q.id, q.status, q.answer, q.attachment_note, (q.attachment_files || []).length]),
          questionReview: job.question_review || {},
          drafts: (job.draft_sections || []).map(section => [section.id, section.title, (section.steps || []).length]),
          draftSlide: state.draftSlideIndexes[job.id] || 0,
          files: (job.files || []).map(file => [file.sha256, file.name, file.processing_status, file.video_status]),
          artifacts: (job.artifacts || []).length,
          transcripts: (state.transcripts || []).map(item => [item.source_file, item.status, item.error, (item.text || '').length]),
          logs: [job.logs?.length || 0, job.logs?.[job.logs.length - 1]?.message || ''],
          needsProcessing: !!job.needs_processing,
          watchdog: job.watchdog || {},
          draftValidation: job.draft_validation || {},
          draftAudit: job.draft_audit || {},
          processingCache: job.processing_cache || {},
          processingHealth: job.processing_health || {}
        });
      }
      if (state.currentPage === 'jobs') {
        return JSON.stringify({
          page: state.currentPage,
          search: state.search,
          showArchived: state.showArchived,
          jobs: state.jobs.map(job => [job.id, job.status, job.progress, job.last_activity, job.file_count, job.questions_pending, job.draft_count, job.needs_processing, job.expected_step_count, job.focus_area])
        });
      }
      if (state.currentPage === 'sections') {
        return JSON.stringify({
          page: state.currentPage,
          search: state.sectionSearch,
          sections: state.sections.map(section => [section.id, section.title, section.status, (section.steps || []).length])
        });
      }
      if (state.currentPage === 'section') {
        const draft = state.sectionDraft;
        return JSON.stringify({
          page: state.currentPage,
          section: state.currentSection,
          draftId: state.sectionDraftId,
          selectedStep: state.sectionStepIndexes[state.currentSection] || 0,
          qaFocus: state.sectionQaFocus,
          draft: draft ? [draft.id, draft.title, draft.status, draft.product, draft.work_area, draft.station, (draft.tags || []).join('|'), (draft.steps || []).map(step => [
            step.step_id,
            (step.media || []).join('|'),
            step.text,
            step.tools,
            step.torque,
            (step.parts || []).join('|'),
            step.quality,
            step.safety,
            step.media_match_confidence,
            !!step.media_fallback,
            !!step.merge_candidate,
            JSON.stringify(step.media_settings || {}),
            JSON.stringify(step.qa_resolutions || {}),
            JSON.stringify(step.evidence_review || {}),
            step.start_time,
            step.end_time,
            step.target_time
          ]).join('||')] : null,
          mediaOptions: (state.sectionMediaOptions[state.currentSection] || []).map(item => [item.path, item.score, item.is_currently_used]),
          mediaPicker: state.sectionMediaPicker
        });
      }
      if (state.currentPage === 'package') {
        return JSON.stringify({
          page: state.currentPage,
          search: state.packageListSearch,
          packages: state.packages.map(packageItem => [packageItem.id, packageItem.title, packageItem.product, packageItem.work_area, packageItem.station, packageItem.status, packageItem.section_count, packageItem.step_count, packageItem.updated_at])
        });
      }
      if (state.currentPage === 'package_edit') {
        const packageData = state.currentPackageData;
        return JSON.stringify({
          page: state.currentPage,
          packageId: state.currentPackage,
          packageData: packageData ? [packageData.id, packageData.title, packageData.status, packageData.updated_at] : null,
          search: state.packageSearch,
          draftId: state.packageDraftId,
          draft: state.packageDraft,
          meta: state.packageDraftMeta,
          expanded: state.expandedPackageSections,
          sections: state.sections.map(section => [section.id, section.title, section.status, (section.steps || []).length]),
        });
      }
      if (state.currentPage === 'package_view' || state.currentPage === 'package_print' || state.currentPage === 'operator_view') {
        const packageData = state.currentPackageData;
        return JSON.stringify({
          page: state.currentPage,
          packageId: state.currentPackage,
          operatorStep: state.operatorStepIndexes[state.currentPackage] || 0,
          packageData: packageData ? [packageData.id, packageData.title, packageData.status, packageData.updated_at, (packageData.sections || []).map(section => [section.id, section.title, (section.steps || []).length])] : null
        });
      }
      return JSON.stringify({page: state.currentPage});
    }

    function renderJobs() {
      setTitle('Jobs');
      setTopActions(`<button class="primary" onclick="openJobModal()">Start Job</button>`);
      const rows = state.jobs
        .filter(j => state.showArchived || j.active)
        .filter(j => {
          const hay = `${j.job_no} ${j.name} ${j.product} ${j.work_area} ${j.station}`.toLowerCase();
          return hay.includes(state.search.toLowerCase());
        });
      document.getElementById('root').innerHTML = `
        <div class="toolbar">
          <div class="toolbar-left">
            <input class="search" placeholder="Search jobs" value="${esc(state.search)}" oninput="state.search=this.value" onblur="renderJobs()">
            <button class="ghost" onclick="state.showArchived=!state.showArchived; renderJobs()">${state.showArchived ? 'Hide Complete' : 'Show Complete'}</button>
          </div>
          <div class="toolbar-right"><span class="subtle">${rows.length} shown</span></div>
        </div>
        <div class="table-wrap" data-scroll-key="jobs-table">
          <table>
            <thead>
              <tr>
                <th>Job</th>
                <th>Context</th>
                <th>Files</th>
                <th>Uploads</th>
                <th>Transcript</th>
                <th>Drawings</th>
                <th>Images</th>
                <th>Video</th>
                <th>Questions</th>
                <th>Drafts</th>
                <th>Overall</th>
                <th>Flags</th>
                <th>Updated</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              ${rows.map(job => `
                <tr>
                  <td class="main-cell">
                    <div class="strong">${esc(job.job_no)} · ${esc(job.name)}</div>
                    <div class="subtle">${statusChip(job.status)} <span class="chip">${esc(job.profile)}</span>${job.needs_processing ? ' <span class="chip warn">Needs processing</span>' : ''}</div>
                  </td>
                  <td>
                    <div>${esc(job.product)}</div>
                    <div class="subtle">${esc(job.work_area)} · ${esc(job.station)}${job.expected_step_count ? ' · ~' + esc(job.expected_step_count) + ' steps' : ''} · ${job.focus_area === 'full' ? 'full frame' : 'center focus'}</div>
                  </td>
                  <td>${job.file_count}</td>
                  <td>${stageCell(job, 'upload_check')}</td>
                  <td>${stageCell(job, 'transcription')}</td>
                  <td>${stageCell(job, 'drawings')}</td>
                  <td>${stageCell(job, 'images')}</td>
                  <td>${stageCell(job, 'video')}</td>
                  <td>${job.questions_pending ? `<span class="chip warn">${job.questions_pending} pending</span>` : stageCell(job, 'questions')}</td>
                  <td>${job.draft_count ? `<span class="chip brand">${job.draft_count} ready</span>` : stageCell(job, 'draft_sections')}</td>
                  <td style="min-width:150px"><div class="bar"><span style="width:${job.progress}%"></span></div><div class="subtle">${job.progress}%</div></td>
                  <td><div class="chip-row">${job.flags.length ? job.flags.map(f => `<span class="chip warn">${esc(f)}</span>`).join('') : '<span class="chip good">None</span>'}</div></td>
                  <td><span class="subtle">${esc(job.last_activity)}</span></td>
                  <td><button onclick="go('job','${job.id}')">Open</button></td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderStageList(job) {
      return `<div class="stage-list">
        ${stageKeys.map(k => {
          const s = job.stages[k];
          const selected = (state.selectedStages[job.id] || activeStageKey(job)) === k;
          return `<div class="stage-item selectable ${selected ? 'active' : ''}" onclick="selectStage('${job.id}', '${k}')">
            ${dot(s)}
            <div>
              <div class="strong">${esc(s.label)}</div>
              <div class="subtle">${esc(s.note || s.status)}</div>
            </div>
            <div class="subtle">${s.percent}%</div>
          </div>`;
        }).join('')}
      </div>`;
    }

    function activeStageKey(job) {
      return stageKeys.find(k => job.stages[k]?.status === 'running')
        || stageKeys.find(k => job.stages[k]?.status === 'blocked' || job.stages[k]?.status === 'failed')
        || (job.draft_sections?.length ? 'draft_sections' : null)
        || (job.questions?.length ? 'questions' : null)
        || 'transcription';
    }

    function selectStage(jobId, key) {
      invalidateRender();
      state.selectedStages[jobId] = key;
      render(false);
    }

    function stageArtifacts(job, kind) {
      return compactArtifacts(job.artifacts || []).filter(file => file.kind === kind);
    }

    function renderFrameGrid(job) {
      const frames = stageArtifacts(job, 'frame');
      if (!frames.length) return `<div class="empty">No frames extracted yet.</div>`;
      return `<div class="frame-grid" data-scroll-key="stage-video-frames">
        ${frames.map(file => {
          const analysis = file.analysis || {};
          const score = analysis.instruction_value ?? file.instruction_value;
          const pointing = analysis.pointing || file.pointing;
          const summary = analysis.visual_summary || analysis.reason || file.processing_status || '';
          const tags = [...new Set([...(file.candidate_tags || []), ...(analysis.tags || [])])].slice(0, 5);
          return `
          <div class="frame-card clickable" onclick="openImageZoom('${jsSingleQuoted(file.stored_path)}', '${jsSingleQuoted(file.timestamp || file.name)}')">
            <img src="${mediaUrl(file.stored_path)}" alt="${esc(file.name)}" loading="lazy">
            <div class="subtle">${esc(file.timestamp || '')} · ${esc(file.name)}</div>
            <div class="chip-row" style="padding:0 6px 6px">
              ${score !== undefined ? `<span class="chip ${score >= 70 ? 'good' : score < 35 ? 'bad' : ''}">Score ${esc(score)}</span>` : ''}
              ${file.local_priority ? `<span class="chip">Local ${esc(file.local_priority)}</span>` : ''}
              ${pointing ? '<span class="chip warn">Pointing</span>' : ''}
              ${tags.map(tag => `<span class="chip">${esc(tag)}</span>`).join('')}
            </div>
            ${summary ? `<div class="subtle" style="padding:0 6px 8px">${esc(summary)}</div>` : ''}
          </div>
        `;
        }).join('')}
      </div>`;
    }

    function renderUploadedImages(job) {
      const images = (job.files || []).filter(file => file.kind === 'image' && !file.removed);
      if (!images.length) {
        return `<div class="empty">No uploaded images yet.</div>`;
      }
      return `<div class="frame-grid" data-scroll-key="stage-uploaded-images">
        ${images.map(file => `
          <div class="frame-card ${isBrowserPreviewImage(file) && file.stored_path ? 'clickable' : ''}" ${isBrowserPreviewImage(file) && file.stored_path ? `onclick="openImageZoom('${jsSingleQuoted(file.stored_path)}', '${jsSingleQuoted(file.name)}')"` : ''}>
            ${isBrowserPreviewImage(file) && file.stored_path
              ? `<img src="${mediaUrl(file.stored_path)}" alt="${esc(file.name)}" loading="lazy">`
              : `<div class="source-tile">${esc((file.name || '').split('.').pop() || 'image')}<br><span class="subtle">Preview unavailable</span></div>`}
            <div class="subtle">${esc(file.name)}</div>
            <div class="chip-row" style="padding:0 6px 8px">
              <span class="chip brand">image</span>
              <span class="chip">${esc(file.size || '')}</span>
              <span class="chip">${esc(file.processing_status || 'stored')}</span>
            </div>
          </div>
        `).join('')}
      </div>`;
    }

    function renderStageDetails(job) {
      const key = state.selectedStages[job.id] || activeStageKey(job);
      const stage = job.stages[key];
      const audio = stageArtifacts(job, 'audio');
      const frames = stageArtifacts(job, 'frame');
      const cueRows = (job.cues || []).map(cue => `<span class="chip">${esc(cue.label)}: ${cue.count}</span>`).join('');
      let body = '';
      if (key === 'upload_check') {
        body = `<div class="stage-detail-list">
          <div class="split-actions" style="justify-content:flex-start;margin-top:0">
            <button class="primary" onclick="openAddFiles('${job.id}')">Add Files</button>
            ${job.needs_processing ? '<span class="chip warn">New files need processing</span>' : '<span class="chip good">Uploads current</span>'}
          </div>
          ${renderUploadedSources(job)}
        </div>`;
      } else if (key === 'transcription') {
        body = `<div class="stage-detail-list">
          <div class="chip-row">
            <span class="chip ${stage.status === 'done' ? 'good' : stage.status === 'failed' ? 'bad' : 'warn'}">${esc(stage.status)}</span>
            <span class="chip">${esc(stage.note || '')}</span>
          </div>
          <div class="chip-row">${cueRows || '<span class="chip">No spoken cues detected yet</span>'}</div>
          <div class="file-list">${audio.map(file => `
            <div class="file-row">
              <div>
                <div class="strong">${esc(file.name)}</div>
                <div class="subtle">${esc(file.type || 'audio')} · ${esc(file.size || '')}</div>
              </div>
              <div class="chip-row"><span class="chip">${esc(file.processing_status || '')}</span></div>
            </div>`).join('') || '<div class="empty">No audio artifact yet.</div>'}</div>
          ${renderTranscripts()}
        </div>`;
      } else if (key === 'video') {
        body = renderFrameGrid(job);
      } else if (key === 'images') {
        const imageCount = (job.files || []).filter(file => file.kind === 'image' && !file.removed).length;
        body = `<div class="stage-detail-list"><div class="chip-row"><span class="chip">${imageCount} uploaded images</span><span class="chip warn">Semantic analysis not implemented</span></div>${renderUploadedImages(job)}</div>`;
      } else if (key === 'drawings') {
        const documents = (job.files || []).filter(file => ['pdf', 'text'].includes(file.kind) && !file.removed);
        const pages = stageArtifacts(job, 'pdf_page');
        const evidence = job.document_evidence || [];
        const evidenceSummary = job.evidence_summary || {};
        const conflicts = (job.evidence_conflicts || []).filter(item => item.status === 'open');
        const applicabilityReviews = new Set(
          (job.evidence_items || [])
            .filter(item => item.evidence_type === 'document_page')
            .filter(item => ['unspecified', 'review_required'].includes(item.applicability_review?.status || 'unspecified'))
            .map(item => item.source_id)
            .filter(Boolean)
        );
        const textChars = evidence.reduce((sum, item) => sum + String(item.text || '').length, 0);
        body = `<div class="stage-detail-list">
          <div class="chip-row">
            <span class="chip">${documents.length} documents</span>
            <span class="chip">${pages.length} rendered pages</span>
            <span class="chip ${textChars ? 'good' : 'warn'}">${textChars} extracted characters</span>
            <span class="chip">${esc(evidenceSummary.fact_count || 0)} technical facts</span>
            <span class="chip ${applicabilityReviews.size ? 'warn' : 'good'}">${applicabilityReviews.size} applicability reviews</span>
            <span class="chip ${conflicts.length ? 'bad' : 'good'}">${conflicts.length} source conflicts</span>
            <span class="chip warn">BOM/title-block interpretation not implemented</span>
          </div>
          <div class="file-list">${documents.map(file => `<div class="file-row"><div><div class="strong">${esc(file.name)}</div><div class="subtle">${esc(file.document_status || 'not processed')} · ${esc(file.document_page_count || 0)} pages · ${esc(file.authority_level || '')}${file.revision ? ' · rev ' + esc(file.revision) : ''}</div></div>${file.document_error ? `<span class="chip bad">${esc(file.document_error)}</span>` : ''}</div>`).join('') || '<div class="empty">No supporting documents attached.</div>'}</div>
          ${conflicts.length ? `<div class="file-list">${conflicts.map(conflict => `<div class="file-row"><div><div class="strong">${esc(String(conflict.fact_type || 'technical').replaceAll('_', ' '))}: ${esc(conflict.subject || 'unspecified subject')}</div><div class="subtle">${(conflict.claims || []).map(claim => `${esc(claim.source_name || 'source')}: ${esc(claim.value || '')}`).join(' · ')}</div></div><span class="chip bad">Review required</span></div>`).join('')}</div>` : ''}
          ${pages.length ? `<div class="frame-grid">${pages.map(page => `<div class="frame-card clickable" onclick="openImageZoom('${jsSingleQuoted(page.stored_path)}', '${jsSingleQuoted((page.source_file || '') + ' page ' + (page.page_number || ''))}')"><img src="${mediaUrl(page.stored_path)}" alt="${esc(page.source_file || 'Document page')}"><div class="subtle">${esc(page.source_file || '')} · page ${esc(page.page_number || '')}<br>${esc(page.summary || '')}</div></div>`).join('')}</div>` : ''}
        </div>`;
      } else if (key === 'questions') {
        body = renderQuestions(job);
      } else if (key === 'draft_sections') {
        body = renderDraftBrowser(job);
      }
      return `<div class="panel stage-detail-card stage-detail-${esc(key)}">
        <div class="panel-head"><h2>${esc(stage.label)} Details</h2><span class="chip">${esc(stage.status)}</span></div>
        <div class="panel-body stage-detail-body" data-scroll-key="stage-detail">${body}</div>
      </div>`;
    }

    function mediaThumb(path, step = null) {
      const value = String(path || '');
      const setting = mediaSettingFor(step || {}, value);
      if (value.match(/\\.(jpg|jpeg|png|webp)$/i) || value.startsWith('storage/')) {
        return `<div class="thumb clickable ${mediaClassNames(setting)}" onclick="openImageZoom('${jsSingleQuoted(value)}', '${jsSingleQuoted(value)}')"><img src="${mediaUrl(value)}" alt="${esc(value)}" loading="lazy"></div>`;
      }
      return `<div class="thumb">${esc(value)}</div>`;
    }

    function mediaDisplayName(path) {
      const value = String(path || '');
      const name = value.split('/').filter(Boolean).pop() || value || 'Media';
      return name.replace(/^[a-f0-9]{8,}_/i, '');
    }

    function mediaCandidateFor(step, path) {
      const value = String(path || '');
      return (step.media_candidates || []).find(item => String(item.path || '') === value) || null;
    }

    function mediaSettingButton(path, key, value, label, selectedValue) {
      const selected = selectedValue === value || (key === 'fit' && value === 'cover' && selectedValue === 'fill');
      return `<button class="${selected ? 'selected' : ''}" onclick="setStepMediaSetting('${jsSingleQuoted(path)}', '${key}', '${value}')">${label}</button>`;
    }

    function renderMediaSettingControls(path, setting) {
      const clean = normalizeMediaSetting(setting);
      return `<div class="media-setting-controls">
        <div class="media-control-row">
          <span class="subtle">Shape</span>
          <span class="segmented">
            ${mediaSettingButton(path, 'aspect', 'auto', 'Auto', clean.aspect)}
            ${mediaSettingButton(path, 'aspect', 'portrait', 'Portrait', clean.aspect)}
            ${mediaSettingButton(path, 'aspect', 'landscape', 'Landscape', clean.aspect)}
            ${mediaSettingButton(path, 'aspect', 'square', 'Square', clean.aspect)}
          </span>
        </div>
        <div class="media-control-row">
          <span class="subtle">Fit</span>
          <span class="segmented">
            ${mediaSettingButton(path, 'fit', 'contain', 'Full', clean.fit)}
            ${mediaSettingButton(path, 'fit', 'cover', 'Fill', clean.fit)}
          </span>
        </div>
        <div class="media-control-row">
          <span class="subtle">Crop</span>
          <span class="segmented">
            ${mediaSettingButton(path, 'crop_position', 'center', 'Center', clean.crop_position)}
            ${mediaSettingButton(path, 'crop_position', 'top', 'Top', clean.crop_position)}
            ${mediaSettingButton(path, 'crop_position', 'bottom', 'Bottom', clean.crop_position)}
            ${mediaSettingButton(path, 'crop_position', 'left', 'Left', clean.crop_position)}
            ${mediaSettingButton(path, 'crop_position', 'right', 'Right', clean.crop_position)}
          </span>
        </div>
      </div>`;
    }

    function mediaCard(step, path, idx, mediaCount) {
      const value = String(path || '');
      const candidate = mediaCandidateFor(step, value);
      const setting = mediaSettingFor(step, value);
      const label = mediaDisplayName(value);
      const previewable = value.match(/\\.(jpg|jpeg|png|webp)$/i) || value.startsWith('storage/');
      const role = idx === 0 ? 'Primary image' : `Support ${idx}`;
      const note = candidate?.summary || candidate?.reason || (idx === 0 ? step.media_fallback_reason : '');
      const confidence = idx === 0 ? step.media_match_confidence : '';
      const score = candidate?.score || (idx === 0 ? step.media_match_score : '');
      const timestamp = candidate?.timestamp_seconds ?? candidate?.timestamp ?? null;
      return `<div class="step-media-card ${idx === 0 ? 'primary' : ''}">
        ${previewable ? `
          <button class="media-card-preview" type="button" onclick="openImageZoom('${jsSingleQuoted(value)}', '${jsSingleQuoted(label)}')">
            <span class="media-display ${mediaClassNames(setting)}"><img src="${mediaUrl(value)}" alt="${esc(label)}" loading="lazy"></span>
            <span class="media-role-badge">${esc(role)}</span>
            <span class="media-zoom-hint">Zoom</span>
          </button>
        ` : `<div class="source-tile">${esc(value)}<br><span class="subtle">Preview unavailable</span></div>`}
        <div class="media-card-body">
          <div class="media-card-title" title="${esc(value)}">${esc(label)}</div>
          <div class="chip-row">
            <span class="chip ${idx === 0 ? 'good' : ''}">${esc(role)}</span>
            ${timestamp !== null && timestamp !== undefined && timestamp !== '' ? `<span class="chip">~${esc(timestamp)}s</span>` : ''}
            ${confidence ? `<span class="chip ${confidence === 'manual' || confidence === 'high' ? 'good' : confidence === 'fallback' || confidence === 'low' ? 'warn' : ''}">${esc(confidence)}</span>` : ''}
            ${score ? `<span class="chip">Score ${esc(score)}</span>` : ''}
          </div>
          ${note ? `<div class="media-card-note">${esc(note)}</div>` : ''}
          ${previewable ? renderMediaSettingControls(value, setting) : ''}
          <div class="media-card-actions">
            <button class="icon" onclick="moveStepMedia(${idx}, -1)" ${idx <= 0 ? 'disabled' : ''} title="Move left">←</button>
            <button class="icon" onclick="moveStepMedia(${idx}, 1)" ${idx >= mediaCount - 1 ? 'disabled' : ''} title="Move right">→</button>
            <button onclick="openSectionMediaPicker(${idx}, 'replace')">Replace</button>
            <button class="danger" onclick="removeStepMedia(${idx})">Remove</button>
          </div>
        </div>
      </div>`;
    }

    function renderQuestionReview(job) {
      const review = job.question_review || {};
      const checks = review.checks || [];
      if (!checks.length) {
        return `<div class="empty compact">Question checks will appear after processing reaches this stage.</div>`;
      }
      return `<div class="question-card ${review.generated_questions ? 'pending' : 'bypassed'}">
        <div class="card-head">
          <div>
            <div class="strong">Question Review</div>
            <div class="subtle">${esc(review.confidence_note || '')}</div>
          </div>
          <span class="chip ${review.generated_questions ? 'warn' : 'good'}">${esc(review.generated_questions || 0)} generated</span>
        </div>
        <div class="stage-detail-list">
          ${checks.map(check => `
            <div class="file-row">
              <div>
                <div class="strong">${esc(check.label)}</div>
                <div class="subtle">${esc(check.detail)}</div>
              </div>
              <span class="chip ${check.result === 'question' ? 'warn' : 'good'}">${esc(check.result)}</span>
            </div>
          `).join('')}
        </div>
      </div>`;
    }

    function renderQuestions(job) {
      const review = renderQuestionReview(job);
      if (!job.questions.length) return `<div class="stack">${review}</div>`;
      return `<div class="stack">
        ${review}
        ${job.questions.map(q => {
          const selected = state.selectedChoices[q.id] ?? q.answer;
          const draftAnswer = state.questionDrafts[q.id] ?? q.answer ?? '';
          const draftNote = state.questionAttachmentNotes[q.id] ?? q.attachment_note ?? '';
          const savedFiles = q.attachment_files || [];
          const draftFiles = state.questionFiles[q.id] || [];
          const shownFiles = draftFiles.length ? draftFiles : savedFiles;
          return `<div class="question-card ${esc(q.status)}">
            <div class="card-head">
              <div>
                <div class="strong">${esc(q.title)}</div>
                <div class="subtle">${esc(q.source)}</div>
              </div>
              <span class="chip ${q.status === 'pending' ? 'warn' : 'good'}">${esc(q.status)}</span>
            </div>
            <div>${esc(q.prompt)}</div>
            <div class="source-tile">${esc(q.source)}</div>
            <div class="subtle">${esc(q.why)}</div>
            <div class="choice-row">
              ${q.choices.map(choice => `<button class="${selected === choice ? 'selected' : ''}" onclick="selectQuestionChoice('${q.id}', '${esc(choice)}')">${esc(choice)}</button>`).join('')}
            </div>
            <textarea id="answer-${q.id}" placeholder="Answer or add context" oninput="state.questionDrafts['${q.id}']=this.value">${esc(draftAnswer)}</textarea>
            <input id="attach-${q.id}" placeholder="Attach note or reference file name" value="${esc(draftNote)}" oninput="state.questionAttachmentNotes['${q.id}']=this.value">
            <input type="file" multiple onchange="captureQuestionFiles('${q.id}', this.files)">
            <div class="chip-row">${shownFiles.length ? shownFiles.map(file => `<span class="chip">${esc(file.name)}</span>`).join('') : '<span class="chip">No clarification files</span>'}</div>
            <div class="split-actions">
              <button onclick="answerQuestion('${q.id}', 'bypassed')">Bypass</button>
              <button class="primary" onclick="answerQuestion('${q.id}', 'answered')">Save Answer</button>
            </div>
          </div>`;
        }).join('')}
      </div>`;
    }

    function optionList(options, selected) {
      return options.map(([value, label]) => `<option value="${esc(value)}" ${value === selected ? 'selected' : ''}>${esc(label)}</option>`).join('');
    }

    function sourceRoleOptions(file) {
      if (file.kind === 'video') {
        return [['primary_video', 'Primary video'], ['alternate_angle', 'Alternate angle']];
      }
      if (file.kind === 'audio') {
        return [['voiceover', 'Voiceover / narration']];
      }
      if (file.kind === 'image') {
        return [['reference_image', 'Reference image'], ['supporting_document', 'Supporting document']];
      }
      if (['pdf', 'text'].includes(file.kind)) {
        return [['supporting_document', 'Supporting document'], ['reference', 'Reference only']];
      }
      return [['reference', 'Reference only'], ['supporting_document', 'Supporting document']];
    }

    function sourceAuthorityOptions() {
      return [
        ['authoritative', 'Authoritative'],
        ['primary_evidence', 'Primary evidence'],
        ['supporting_context', 'Supporting context'],
        ['reference_only', 'Reference only']
      ];
    }

    function sourceSetupId(file) {
      return String(file.source_id || uploadIdentity(file) || '').replace(/[^A-Za-z0-9_-]+/g, '_');
    }

    function sourceAlignmentSummary(file) {
      const alignment = file.timeline_alignment || {};
      if (file.timeline_reference) return 'Reference timeline';
      if (alignment.method === 'manual_anchors') {
        const drift = alignment.has_drift_anchor ? ` · drift scale ${Number(alignment.scale || 1).toFixed(6)}` : '';
        return `Manual · ${Number(alignment.offset_seconds || 0).toFixed(2)}s offset${drift}`;
      }
      if (alignment.method === 'transcript_word_match') {
        return `Auto · ${Number(alignment.offset_seconds || 0).toFixed(2)}s offset · ${esc(alignment.confidence || 'unknown')}`;
      }
      if (alignment.method === 'audio_waveform') {
        return `Shared audio · ${Number(alignment.offset_seconds || 0).toFixed(2)}s offset · ${esc(alignment.confidence || 'unknown')}`;
      }
      return alignment.reason || 'Alignment will be calculated during transcription.';
    }

    function renderSourceSetup(job, file) {
      const domId = sourceSetupId(file);
      const timed = ['video', 'audio'].includes(file.kind);
      const sync = file.timeline_sync || {};
      const openKey = `${job.id}:${file.source_id || uploadIdentity(file)}`;
      const isOpen = !!state.sourceSetupOpen[openKey];
      return `<details class="source-setup" ${isOpen ? 'open' : ''} ontoggle="state.sourceSetupOpen['${jsSingleQuoted(openKey)}']=this.open">
        <summary>Source setup</summary>
        <div class="source-setup-grid">
          <div class="field"><label>Role</label><select id="source-role-${domId}">${optionList(sourceRoleOptions(file), file.source_role)}</select></div>
          <div class="field"><label>Label / camera</label><input id="source-label-${domId}" value="${esc(file.source_label || '')}" placeholder="Overhead, left side, clean voiceover"></div>
          <div class="field"><label>Authority</label><select id="source-authority-${domId}">${optionList(sourceAuthorityOptions(), file.authority_level)}</select></div>
          <div class="field"><label>Revision</label><input id="source-revision-${domId}" value="${esc(file.revision || '')}" placeholder="Optional"></div>
          <div class="field source-applicability"><label>Applicability</label><input id="source-applicability-${domId}" value="${esc(file.applicability || '')}" placeholder="Model, serial range, configuration, or purpose"></div>
        </div>
        ${timed ? `<div class="source-timeline">
          <label class="source-choice"><input id="source-narration-${domId}" type="radio" name="source-narration-${job.id}" ${file.use_for_drafting ? 'checked' : ''}> Use this source for instruction narration</label>
          <label class="source-choice"><input id="source-reference-${domId}" type="radio" name="source-reference-${job.id}" ${file.timeline_reference ? 'checked' : ''}> Use as reference timeline</label>
          <div class="subtle source-alignment-summary">${sourceAlignmentSummary(file)}</div>
          <div class="source-sync-grid">
            <div class="field"><label>Sync method</label><select id="source-sync-mode-${domId}" ${file.timeline_reference ? 'disabled' : ''} onchange="toggleSourceSyncFields('${domId}')">
              <option value="reference" ${file.timeline_reference ? 'selected' : ''}>Reference</option>
              <option value="auto" ${!file.timeline_reference && sync.mode !== 'manual' ? 'selected' : ''}>Automatic</option>
              <option value="manual" ${!file.timeline_reference && sync.mode === 'manual' ? 'selected' : ''}>Manual anchors</option>
            </select></div>
            <div class="field source-manual-field source-manual-${domId} ${sync.mode === 'manual' && !file.timeline_reference ? '' : 'hidden'}"><label>Start: source seconds</label><input id="source-start-source-${domId}" type="number" step="0.01" value="${esc(sync.start_source_seconds ?? '')}"></div>
            <div class="field source-manual-field source-manual-${domId} ${sync.mode === 'manual' && !file.timeline_reference ? '' : 'hidden'}"><label>Start: timeline seconds</label><input id="source-start-timeline-${domId}" type="number" step="0.01" value="${esc(sync.start_timeline_seconds ?? '')}"></div>
            <div class="field source-manual-field source-manual-${domId} ${sync.mode === 'manual' && !file.timeline_reference ? '' : 'hidden'}"><label>End: source seconds</label><input id="source-end-source-${domId}" type="number" step="0.01" value="${esc(sync.end_source_seconds ?? '')}" placeholder="Optional drift anchor"></div>
            <div class="field source-manual-field source-manual-${domId} ${sync.mode === 'manual' && !file.timeline_reference ? '' : 'hidden'}"><label>End: timeline seconds</label><input id="source-end-timeline-${domId}" type="number" step="0.01" value="${esc(sync.end_timeline_seconds ?? '')}" placeholder="Optional drift anchor"></div>
          </div>
        </div>` : ''}
        <div class="split-actions source-setup-actions">
          <button onclick="saveSourceSetup('${job.id}', '${jsSingleQuoted(file.source_id || uploadIdentity(file))}', '${domId}')">Save Setup</button>
          <button class="danger" onclick="removeUpload('${job.id}', '${jsSingleQuoted(uploadIdentity(file))}')">Remove</button>
        </div>
      </details>`;
    }

    function renderUploadedSources(job) {
      const files = (job.files || []).filter(file => !file.removed);
      if (!files.length) {
        return `<div class="empty">No uploaded sources yet.</div>`;
      }
      return `<div class="file-list source-file-list">
        ${files.map(file => `
          <div class="file-row source-file-row">
            <div class="source-file-summary">
              <div class="strong">${esc(file.name)}</div>
              <div class="subtle">${esc(file.type || 'unknown')} · ${esc(file.size || '')}</div>
              <div class="chip-row">
                ${file.kind ? `<span class="chip brand">${esc(file.kind)}</span>` : ''}
                ${file.source_role ? `<span class="chip">${esc(file.source_role.replaceAll('_', ' '))}</span>` : ''}
                ${file.source_label ? `<span class="chip">${esc(file.source_label)}</span>` : ''}
                ${file.use_for_drafting ? '<span class="chip good">Narration</span>' : ''}
                ${file.timeline_reference ? '<span class="chip good">Reference timeline</span>' : ''}
                ${file.authority_level ? `<span class="chip">${esc(file.authority_level.replaceAll('_', ' '))}</span>` : ''}
                ${file.processing_status ? `<span class="chip">${esc(file.processing_status)}</span>` : ''}
                ${file.video_status ? `<span class="chip">${esc(file.video_status)}</span>` : ''}
                ${file.clarification_for ? `<span class="chip warn">${esc(file.clarification_for)}</span>` : ''}
              </div>
            </div>
            ${renderSourceSetup(job, file)}
          </div>
        `).join('')}
      </div>`;
    }

    function transcriptStatusClass(status) {
      if (status === 'complete') return 'good';
      if (status === 'transcription_failed' || status === 'read_error' || status === 'missing_file') return 'bad';
      return 'warn';
    }

    function renderTranscripts() {
      const transcripts = state.transcripts || [];
      if (!transcripts.length) {
        return `<div class="empty">Transcripts will appear after audio is extracted and sent to the transcription provider.</div>`;
      }
      const draftingTranscripts = transcripts.filter(item => item.use_for_drafting);
      const instructionSources = draftingTranscripts.length ? draftingTranscripts : transcripts.slice(0, 1);
      const combinedText = instructionSources.map(item => item.text || '').filter(Boolean).join('\\n\\n');
      return `<div class="step-card">
        <div class="card-head">
          <div>
            <div class="strong">Instruction Transcript</div>
            <div class="subtle">Drafting from ${instructionSources.map(item => esc(item.source_label || item.source_file)).join(', ')}. Alternate transcripts are retained for synchronization only.</div>
          </div>
          <span class="chip ${transcripts.every(item => item.status === 'complete') ? 'good' : 'warn'}">${transcripts.filter(item => item.status === 'complete').length}/${transcripts.length} complete</span>
        </div>
        <div class="chip-row" style="margin-bottom:8px">
          ${transcripts.map(item => `<span class="chip ${transcriptStatusClass(item.status)}">${esc(item.source_label || item.source_file)}: ${esc(item.status)}${item.use_for_drafting ? ' · narration' : ''}${item.timeline_reference ? ' · reference' : ''}</span>`).join('')}
        </div>
        <div class="chip-row" style="margin-bottom:8px">
          ${transcripts.map(item => `<span class="chip">${esc(item.source_file)}: ${esc((item.words || []).length)} timed words</span>`).join('')}
          ${state.currentJobData?.step_parse?.rough_step_count ? `<span class="chip good">${esc(state.currentJobData.step_parse.rough_step_count)} rough steps</span>` : ''}
          ${state.currentJobData?.step_parse?.expected_step_count ? `<span class="chip">Expected ${esc(state.currentJobData.step_parse.expected_step_count)}</span>` : ''}
        </div>
        ${transcripts.some(item => item.error) ? `<div class="chip-row" style="margin-bottom:8px">${transcripts.filter(item => item.error).map(item => `<span class="chip bad" style="white-space:normal">${esc(item.source_file)}: ${esc(item.error)}</span>`).join('')}</div>` : ''}
        ${transcripts.flatMap(item => item.chunks || []).length ? `<div class="chip-row" style="margin-bottom:8px">${transcripts.flatMap(item => item.chunks || []).map((chunk, idx) => `<span class="chip">Chunk ${idx + 1}: ${esc(chunk.size || '')}</span>`).join('')}</div>` : ''}
        <textarea readonly class="transcript-box" data-autogrow="true" data-max-height="560" data-scroll-key="combined-transcript">${esc(combinedText || 'No transcript text yet.')}</textarea>
      </div>`;
    }

    function draftSlides(job) {
      const slides = [];
      (job.draft_sections || []).forEach((section, sectionIndex) => {
        (section.steps || []).forEach((step, stepIndex) => {
          slides.push({section, sectionIndex, step, stepIndex});
        });
      });
      return slides;
    }

    function setDraftSlide(jobId, index) {
      const job = state.currentJobData;
      const total = draftSlides(job).length;
      if (!total) return;
      invalidateRender();
      state.draftSlideIndexes[jobId] = Math.max(0, Math.min(total - 1, index));
      render(false);
    }

    function renderDraftAudit(job) {
      const validation = job.draft_validation || {};
      const audit = job.draft_audit || {};
      if (!audit.steps && !validation.issue_count) return '';
      return `<div class="draft-audit chip-row">
        ${audit.steps ? `<span class="chip">Audit: ${esc(audit.sections || 0)} sections · ${esc(audit.steps || 0)} steps</span>` : ''}
        <span class="chip ${validation.issue_count ? 'warn' : 'good'}">${esc(validation.issue_count || 0)} QA items</span>
        ${(job.media_selection?.fallback_image_steps || 0) ? `<span class="chip warn">${esc(job.media_selection.fallback_image_steps)} fallback images</span>` : ''}
        ${(job.media_selection?.zero_image_steps || 0) ? `<span class="chip bad">${esc(job.media_selection.zero_image_steps)} no image</span>` : ''}
        ${(job.media_selection?.merge_candidate_steps || 0) ? `<span class="chip warn">${esc(job.media_selection.merge_candidate_steps)} merge candidates</span>` : ''}
      </div>`;
    }

    function moveDraftSlide(jobId, delta) {
      const current = state.draftSlideIndexes[jobId] || 0;
      setDraftSlide(jobId, current + delta);
    }

    function renderDraftBrowser(job) {
      const slides = draftSlides(job);
      if (!slides.length) return `<div class="empty">Draft sections will appear after questions are resolved.</div>`;
      const index = Math.max(0, Math.min(slides.length - 1, state.draftSlideIndexes[job.id] || 0));
      state.draftSlideIndexes[job.id] = index;
      const slide = slides[index];
      const section = slide.section;
      const step = slide.step;
      const sectionStepCount = section.steps.length;
      return `<div class="draft-browser" data-scroll-key="draft-browser">
        <div class="draft-browser-head">
          <div>
            <div class="strong">${esc(section.title)}</div>
            <div class="subtle">Section ${slide.sectionIndex + 1} of ${job.draft_sections.length} · Step ${slide.stepIndex + 1} of ${sectionStepCount}</div>
          </div>
          <div class="chip-row">
            <span class="chip">${index + 1}/${slides.length}</span>
            <span class="chip">${esc(job.draft_source || 'draft')}</span>
            ${job.step_parse?.rough_step_count ? `<span class="chip">${esc(job.step_parse.rough_step_count)} parsed steps</span>` : ''}
            ${job.media_selection?.scored_frames ? `<span class="chip good">${esc(job.media_selection.steps || 0)} media matches</span>` : ''}
            ${job.media_selection?.pointing_frames ? `<span class="chip warn">${esc(job.media_selection.pointing_frames)} pointing frames</span>` : ''}
          </div>
        </div>
        ${renderDraftAudit(job)}
        <div class="draft-step-view">
          <div class="draft-step-text">${esc(step.text)}</div>
          <div class="chip-row">
            <span class="chip">Tool: ${esc(step.tools || 'None')}</span>
            <span class="chip ${step.torque && step.torque !== 'Not required' ? 'warn' : ''}">Torque: ${esc(step.torque || 'None')}</span>
            ${step.timing_confidence ? `<span class="chip">Timing: ${esc(step.timing_confidence)}</span>` : ''}
            ${step.media_match_confidence ? `<span class="chip ${step.media_match_confidence === 'high' ? 'good' : step.media_match_confidence === 'missing' ? 'bad' : ''}">Media: ${esc(step.media_match_confidence)}</span>` : ''}
            ${(step.media_candidates || []).length ? `<span class="chip">${esc((step.media_candidates || []).length)} alternates</span>` : ''}
            ${step.quality ? `<span class="chip good">QC: ${esc(step.quality)}</span>` : ''}
            ${step.safety ? `<span class="chip warn">Safety: ${esc(step.safety)}</span>` : ''}
          </div>
          ${(step.parts || []).length ? `<div class="chip-row">${step.parts.map(part => `<span class="chip brand">${esc(part)}</span>`).join('')}</div>` : ''}
          <div class="draft-media">${(step.media || []).map(m => mediaThumb(m, step)).join('') || '<div class="thumb">No media</div>'}</div>
        </div>
        <div class="draft-nav">
          <button onclick="moveDraftSlide('${job.id}', -1)" ${index <= 0 ? 'disabled' : ''}>&larr; Previous</button>
          <div class="chip-row">
            ${job.draft_sections.map((draftSection, sectionIndex) => {
              const startIndex = slides.findIndex(item => item.sectionIndex === sectionIndex);
              return `<button class="${sectionIndex === slide.sectionIndex ? 'primary' : ''}" onclick="setDraftSlide('${job.id}', ${startIndex})">${esc(draftSection.title)}</button>`;
            }).join('')}
          </div>
          <button onclick="moveDraftSlide('${job.id}', 1)" ${index >= slides.length - 1 ? 'disabled' : ''}>Next &rarr;</button>
        </div>
      </div>`;
    }

    function renderLog(job) {
      return `<div class="log">${job.logs.slice().reverse().map(line => `<div class="log-line"><span>${esc(line.time)}</span><span>${esc(line.message)}</span></div>`).join('')}</div>`;
    }

    function renderJobReliability(job) {
      const validation = job.draft_validation || {};
      const audit = job.draft_audit || {};
      const cache = job.processing_cache || {};
      const watchdog = job.watchdog || {};
      const health = job.processing_health || {};
      return `<div class="step-card">
        <div class="card-head">
          <div>
            <div class="strong">Reliability</div>
            <div class="subtle">Watchdog, timing health, cache reuse, and draft audit</div>
          </div>
          <span class="chip ${(validation.issue_count || (health.warnings || []).length) ? 'warn' : 'good'}">${esc((validation.issue_count || 0) + ((health.warnings || []).length || 0))} items</span>
        </div>
        <div class="chip-row">
          ${watchdog.stage ? `<span class="chip ${watchdog.thread_alive ? '' : 'warn'}">Watchdog: ${esc(watchdog.stage)} · ${esc(Math.round((watchdog.idle_seconds || 0) / 60))}m idle</span>` : '<span class="chip good">Watchdog clear</span>'}
          <span class="chip ${health.timed_words ? 'good' : (health.transcript_sources ? 'warn' : '')}">${esc(health.timed_words || 0)} timed words</span>
          ${health.timing_steps ? `<span class="chip ${health.timing_matched_steps === health.timing_steps ? 'good' : 'warn'}">Timing ${esc(health.timing_matched_steps || 0)}/${esc(health.timing_steps || 0)}</span>` : ''}
          ${health.rough_steps ? `<span class="chip">Parsed ${esc(health.rough_steps)} steps</span>` : ''}
          ${health.zero_image_steps ? `<span class="chip bad">${esc(health.zero_image_steps)} no image</span>` : ''}
          ${health.fallback_image_steps ? `<span class="chip warn">${esc(health.fallback_image_steps)} fallback media</span>` : ''}
          ${cache.transcription ? `<span class="chip">Transcript cache: ${esc(cache.transcription.transcripts_reused || 0)} reused</span>` : ''}
          ${cache.video ? `<span class="chip">Video cache: ${esc(cache.video.raw_frame_sets_reused || 0)} frame sets reused</span>` : ''}
          ${cache.frame_analysis ? `<span class="chip">Frame analysis: ${esc(cache.frame_analysis.analysis_reused || 0)} reused</span>` : ''}
          ${audit.steps ? `<span class="chip">Audit: ${esc(audit.sections || 0)} sections · ${esc(audit.steps || 0)} steps</span>` : ''}
        </div>
        ${(health.warnings || []).length ? `<div class="chip-row" style="margin-top:8px">${health.warnings.map(item => `<span class="chip warn">${esc(item)}</span>`).join('')}</div>` : ''}
        ${validation.issue_counts ? `<div class="chip-row" style="margin-top:8px">${Object.entries(validation.issue_counts).map(([key, count]) => `<span class="chip ${count ? 'warn' : ''}">${esc(key)}: ${esc(count)}</span>`).join('')}</div>` : ''}
      </div>`;
    }

    function renderJob() {
      const job = state.currentJobData;
      if (!job) {
        document.getElementById('root').innerHTML = `<div class="empty">Job not found.</div>`;
        return;
      }
      setTitle('Job Review');
      const processingActive = ['Processing','Questions Pending','Drafting Sections'].includes(job.status);
      const canStartProcessing = (!job.processing_started || job.needs_processing) && !processingActive;
      setTopActions(`
        <button onclick="go('jobs')">Back</button>
        <button onclick="startJob('${job.id}')" ${canStartProcessing ? '' : 'disabled'}>Start Processing</button>
        <button class="primary" onclick="completeJob('${job.id}')" ${job.draft_sections.length ? '' : 'disabled'}>Mark Complete</button>
      `);
      const advancedOpen = Object.prototype.hasOwnProperty.call(state.advancedOpenJobs, job.id)
        ? state.advancedOpenJobs[job.id]
        : ['Processing', 'Questions Pending', 'Drafting Sections'].includes(job.status);
      document.getElementById('root').innerHTML = `
        <div class="toolbar">
          <div>
            <div class="strong">${esc(job.job_no)} · ${esc(job.name)}</div>
            <div class="subtle">${esc(job.product)} · ${esc(job.work_area)} · ${esc(job.station)}${job.expected_step_count ? ' · ~' + esc(job.expected_step_count) + ' expected steps' : ''} · ${job.focus_area === 'full' ? 'full frame' : 'center focus'}</div>
          </div>
          <div class="chip-row">${statusChip(job.status)}<span class="chip">${esc(job.profile)}</span>${job.flags.map(f => `<span class="chip warn">${esc(f)}</span>`).join('')}</div>
        </div>
        <div class="stage-review-grid">
          <div class="panel">
            <div class="panel-head"><h2>Progress</h2><span class="subtle">${job.progress}%</span></div>
            <div class="panel-body stack" data-scroll-key="section-library">
              <div class="bar"><span style="width:${job.progress}%"></span></div>
              ${renderStageList(job)}
            </div>
          </div>
          ${renderStageDetails(job)}
        </div>
        <details ${advancedOpen ? 'open' : ''} ontoggle="state.advancedOpenJobs['${job.id}']=this.open">
          <summary>Advanced</summary>
          <div class="details-body stack">
            <div class="split-actions" style="justify-content:flex-start">
              <button onclick="openAddFiles('${job.id}')">Add Files</button>
              <button onclick="resumeCurrentStage('${job.id}')" ${processingActive ? 'disabled' : ''}>Resume Current</button>
              <button onclick="rerunJob('${job.id}')" ${processingActive ? 'disabled' : ''}>Rerun Job</button>
              <button onclick="rerunStage('${job.id}', 'transcription')" ${processingActive ? 'disabled' : ''}>Rerun Transcript</button>
              <button onclick="rerunStage('${job.id}', 'drawings')" ${processingActive ? 'disabled' : ''}>Rerun Documents</button>
              <button onclick="rerunStage('${job.id}', 'video')" ${processingActive ? 'disabled' : ''}>Rerun Video</button>
              <button onclick="rerunStage('${job.id}', 'draft_sections')" ${processingActive ? 'disabled' : ''}>Rerun Drafts</button>
              <button onclick="rerunStage('${job.id}', 'media_assignment')" ${processingActive || !job.draft_sections.length ? 'disabled' : ''}>Rerun Media</button>
              <button class="danger" onclick="deleteJob('${job.id}')" ${processingActive ? 'disabled' : ''}>Delete Job</button>
            </div>
            ${renderJobReliability(job)}
            ${renderLog(job)}
          </div>
        </details>
      `;
    }

    function sectionSearchText(section) {
      const steps = (section.steps || []).map(step => [
        step.text,
        step.tools,
        step.torque,
        step.quality,
        step.safety,
        (step.parts || []).join(' '),
        (step.media || []).join(' ')
      ].join(' ')).join(' ');
      return [
        section.title,
        section.product,
        section.work_area,
        section.station,
        (section.tags || []).join(' '),
        section.source_job,
        steps
      ].join(' ').toLowerCase();
    }

    function filterSectionRows() {
      const query = String(state.sectionSearch || '').toLowerCase();
      let shown = 0;
      document.querySelectorAll('[data-section-row]').forEach(row => {
        const match = (row.dataset.sectionSearch || '').includes(query);
        row.style.display = match ? '' : 'none';
        if (match) shown += 1;
      });
      const count = document.getElementById('section-count');
      if (count) count.textContent = `${shown} sections`;
      const empty = document.getElementById('section-empty-row');
      if (empty) empty.style.display = shown ? 'none' : '';
    }

    function renderSections() {
      setTitle('Sections');
      setTopActions('');
      const rows = state.sections;
      document.getElementById('root').innerHTML = `
        <div class="toolbar">
          <div class="toolbar-left"><input class="search" placeholder="Search sections" value="${esc(state.sectionSearch)}" oninput="state.sectionSearch=this.value; filterSectionRows()"></div>
          <div class="toolbar-right"><span class="subtle" id="section-count">${rows.length} sections</span></div>
        </div>
        <div class="table-wrap" data-scroll-key="sections-table">
          <table class="section-table">
            <thead>
              <tr>
                <th>Status</th>
                <th>Title</th>
                <th>Product</th>
                <th>Work Area</th>
                <th>Station</th>
                <th>Tags</th>
                <th>Steps</th>
                <th>Source Job</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              ${rows.map(s => `
                <tr data-section-row data-section-search="${esc(sectionSearchText(s))}">
                  <td>${statusChip(s.status || 'Needs Edit')}</td>
                  <td class="section-title-cell">
                    <div class="strong">${esc(s.title)}</div>
                    <div class="subtle">${esc((s.steps || [])[0]?.text || '')}</div>
                  </td>
                  <td>${esc(s.product || '')}</td>
                  <td>${esc(s.work_area || '')}</td>
                  <td>${esc(s.station || '')}</td>
                  <td><div class="chip-row">${(s.tags || []).map(t => `<span class="chip">${esc(t)}</span>`).join('') || '<span class="subtle">None</span>'}</div></td>
                  <td>${(s.steps || []).length}</td>
                  <td><span class="subtle">${esc(s.source_job || '')}</span></td>
                  <td>
                    <div class="section-actions">
                      <button onclick="go('section','${s.id}')">Edit</button>
                      <button onclick="duplicateSection('${s.id}')">Duplicate</button>
                      <button onclick="markSectionComplete('${s.id}')" ${s.status === 'Complete' ? 'disabled' : ''}>Mark Complete</button>
                      <button class="primary" onclick="addToPackage('${s.id}')" ${s.status === 'Complete' ? '' : 'disabled'}>Add to Package</button>
                      <button class="danger" onclick="deleteSection('${s.id}')">Delete</button>
                    </div>
                  </td>
                </tr>
              `).join('')}
              <tr id="section-empty-row" style="display:none"><td colspan="9"><div class="empty">No matching sections.</div></td></tr>
            </tbody>
          </table>
        </div>
      `;
      filterSectionRows();
    }

    function cloneSection(section) {
      return JSON.parse(JSON.stringify(section));
    }

    function newStepId() {
      if (window.crypto?.randomUUID) return `step-${window.crypto.randomUUID().slice(0, 8)}`;
      return `step-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    }

    function blankStep() {
      return {
        step_id: newStepId(),
        text: '',
        tools: '',
        torque: '',
        quality: '',
        safety: '',
        parts: [],
        media: [],
        media_settings: {},
        qa_resolutions: {}
      };
    }

    function ensureStepIdentity(step) {
      if (!step.step_id) step.step_id = newStepId();
      if (!step.qa_resolutions || typeof step.qa_resolutions !== 'object' || Array.isArray(step.qa_resolutions)) {
        step.qa_resolutions = {};
      }
      step.parts = normalizeMediaList(step.parts || []);
      step.media = normalizeMediaList(step.media || []);
      step.media_settings = normalizeStepMediaSettings(step);
      return step;
    }

    function ensureDraftStepIds(draft) {
      if (!draft) return draft;
      if (!draft.steps || !draft.steps.length) draft.steps = [blankStep()];
      draft.steps.forEach(ensureStepIdentity);
      return draft;
    }

    function ensureSectionDraft(section) {
      if (!section) return null;
      if (state.sectionDraftId !== section.id || !state.sectionDraft) {
        state.sectionDraftId = section.id;
        state.sectionDraft = cloneSection(section);
        if (!state.sectionDraft.steps || !state.sectionDraft.steps.length) {
          state.sectionDraft.steps = [blankStep()];
        }
        ensureDraftStepIds(state.sectionDraft);
        state.sectionStepIndexes[section.id] = 0;
      }
      return state.sectionDraft;
    }

    function commaList(value) {
      return String(value || '').split(',').map(item => item.trim()).filter(Boolean);
    }

    function tokenValues(value) {
      return uniqueList(Array.isArray(value) ? value : commaList(value));
    }

    function tokenValueText(values) {
      return tokenValues(values).join(', ');
    }

    function renderTokenBadges(id, values) {
      return tokenValues(values).map((value, idx) => `
        <button type="button" class="token-badge" onclick="removeTokenValue('${id}', ${idx})" title="Remove ${esc(value)}">
          <span>${esc(value)}</span><span class="token-remove">&times;</span>
        </button>
      `).join('');
    }

    function tokenField(id, value, placeholder = 'Add item') {
      const values = tokenValues(value);
      return `<div class="token-field" data-token-field="${id}">
        <input type="hidden" id="${id}" value="${esc(tokenValueText(values))}">
        <span class="token-list" id="${id}-tokens">${renderTokenBadges(id, values)}</span>
        <input class="token-input" id="${id}-entry" placeholder="${esc(placeholder)}" onkeydown="handleTokenFieldKey(event, '${id}')" onblur="commitTokenFieldInput('${id}')">
      </div>`;
    }

    function getTokenValues(id) {
      return tokenValues(document.getElementById(id)?.value || '');
    }

    function setTokenValues(id, values) {
      const clean = tokenValues(values);
      const hidden = document.getElementById(id);
      if (hidden) hidden.value = tokenValueText(clean);
      const tokens = document.getElementById(`${id}-tokens`);
      if (tokens) tokens.innerHTML = renderTokenBadges(id, clean);
    }

    function commitTokenFieldInput(id) {
      const entry = document.getElementById(`${id}-entry`);
      if (!entry) return;
      const additions = commaList(entry.value);
      if (!additions.length) return;
      setTokenValues(id, [...getTokenValues(id), ...additions]);
      entry.value = '';
    }

    function handleTokenFieldKey(event, id) {
      if (event.key === 'Enter' || event.key === ',') {
        event.preventDefault();
        commitTokenFieldInput(id);
        return;
      }
      if (event.key === 'Backspace') {
        const entry = document.getElementById(`${id}-entry`);
        if (entry && !entry.value) {
          const values = getTokenValues(id);
          values.pop();
          setTokenValues(id, values);
        }
      }
    }

    function removeTokenValue(id, index) {
      const values = getTokenValues(id);
      values.splice(index, 1);
      setTokenValues(id, values);
      const entry = document.getElementById(`${id}-entry`);
      if (entry) entry.focus();
    }

    function normalizeMediaList(value) {
      if (Array.isArray(value)) return value.map(item => String(item || '').trim()).filter(Boolean);
      return commaList(value);
    }

    function uniqueList(items) {
      const seen = new Set();
      return (items || []).map(item => String(item || '').trim()).filter(Boolean).filter(item => {
        const key = item.toLowerCase();
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
    }

    function combineTextValues(values, separator = ' ') {
      return uniqueList(values).join(separator).replace(/\s+/g, ' ').trim();
    }

    function qaResolutionHandled(step, issueId) {
      const status = step?.qa_resolutions?.[issueId]?.status;
      return status === 'bypassed' || status === 'resolved';
    }

    function setQaResolution(step, issueId, status = 'bypassed', note = '') {
      ensureStepIdentity(step);
      step.qa_resolutions[issueId] = {
        status,
        note,
        at: new Date().toISOString()
      };
    }

    function qaIssueDefinition(issueId) {
      return {
        blank_text: {label: 'Blank instruction', detail: 'Instruction text is empty.', target: 'step-text'},
        missing_media: {label: 'Missing media', detail: 'No image is attached to this step.', target: 'media'},
        fallback_media: {label: 'Low-confidence media', detail: 'Image was selected as fallback or scored low.', target: 'media'},
        merge_candidate: {label: 'Merge candidate', detail: 'This step may be too small to stand alone.', target: 'merge'},
        missing_parts: {label: 'Missing parts', detail: 'Step wording suggests parts or consumables, but none are listed.', target: 'step-parts'},
        missing_torque: {label: 'Missing torque', detail: 'Fastener wording suggests torque may matter.', target: 'step-torque'},
        quality_cue: {label: 'Quality cue', detail: 'Step wording suggests a check, but QC is empty.', target: 'step-quality'},
        safety_cue: {label: 'Safety cue', detail: 'Step wording suggests a hazard, but safety is empty.', target: 'step-safety'}
      }[issueId] || {label: issueId, detail: '', target: ''};
    }

    function stepQaIssues(step, idx) {
      ensureStepIdentity(step);
      const text = String(step.text || '').trim();
      const lower = text.toLowerCase();
      const media = normalizeMediaList(step.media);
      const parts = normalizeMediaList(step.parts);
      const torque = String(step.torque || '').trim().toLowerCase();
      const issues = [];
      const add = issueId => {
        if (qaResolutionHandled(step, issueId)) return;
        const def = qaIssueDefinition(issueId);
        issues.push({issueId, stepIndex: idx, stepId: step.step_id, label: def.label, detail: def.detail, target: def.target});
      };
      if (!text) add('blank_text');
      if (!media.length) add('missing_media');
      else if (step.media_fallback || ['fallback', 'low', 'missing'].includes(step.media_match_confidence)) add('fallback_media');
      if (step.merge_candidate) add('merge_candidate');
      if (lower && !parts.length && /\b(attach|install|insert|place|apply|assemble|add|mount|connect|bolt|screw|fasten|position|weld)\b/.test(lower)) add('missing_parts');
      if (/\b(bolt|screw|fastener|nut|tighten|torque)\b/.test(lower) && (!torque || ['none', 'not required', 'n/a', 'na'].includes(torque))) add('missing_torque');
      if (!String(step.quality || '').trim() && /\b(check|verify|confirm|inspect|test|measure|flush|seated|alignment|align)\b/.test(lower)) add('quality_cue');
      if (!String(step.safety || '').trim() && /\b(sharp|hot|pinch|ppe|glove|weld|paint|fume|disconnect|lockout|lift|heavy|pressure|energized)\b/.test(lower)) add('safety_cue');
      return issues;
    }

    function sectionQaIssues(draft) {
      ensureDraftStepIds(draft);
      return (draft.steps || []).flatMap((step, idx) => stepQaIssues(step, idx));
    }

    function selectedSectionStepIndex(draft) {
      const count = Math.max((draft?.steps || []).length, 1);
      const raw = state.sectionStepIndexes[draft.id] ?? 0;
      const index = Math.min(Math.max(Number(raw) || 0, 0), count - 1);
      state.sectionStepIndexes[draft.id] = index;
      return index;
    }

    function selectedSectionStep() {
      const draft = collectSectionDraft();
      if (!draft) return null;
      return draft.steps[selectedSectionStepIndex(draft)] || null;
    }

    function openSectionMediaPicker(replaceIndex = null, mode = 'append') {
      const draft = collectSectionDraft();
      if (!draft) return;
      state.sectionMediaPicker = {
        open: true,
        replaceIndex,
        mode,
        filter: state.sectionMediaPicker.filter || 'near',
        search: state.sectionMediaPicker.search || ''
      };
      renderSectionEditor();
    }

    function closeSectionMediaPicker() {
      state.sectionMediaPicker.open = false;
      state.sectionMediaPicker.replaceIndex = null;
      state.sectionMediaPicker.mode = 'append';
      renderSectionEditor();
    }

    function setSectionMediaPickerFilter(filter) {
      state.sectionMediaPicker.filter = filter;
      renderSectionEditor();
    }

    function setSectionMediaPickerSearch(value) {
      state.sectionMediaPicker.search = value;
      filterSectionMediaOptionCards();
    }

    function filterSectionMediaOptionCards() {
      const query = String(state.sectionMediaPicker.search || '').toLowerCase();
      let shown = 0;
      document.querySelectorAll('[data-media-option]').forEach(card => {
        const match = (card.dataset.mediaSearch || '').includes(query);
        card.style.display = match ? '' : 'none';
        if (match) shown += 1;
      });
      const count = document.getElementById('media-picker-count');
      if (count) count.textContent = `${shown} options shown`;
      const empty = document.getElementById('media-picker-empty');
      if (empty) empty.style.display = shown ? 'none' : '';
    }

    function clearStepMediaFallback(step, manual = true) {
      step.media_fallback = false;
      step.media_fallback_reason = '';
      if (manual) step.media_match_confidence = 'manual';
      step.evidence_review = {
        ...(step.evidence_review || {}),
        media_reviewed: true,
        image_manually_selected: manual ? true : !!step.evidence_review?.image_manually_selected,
        updated_at: new Date().toISOString()
      };
      setQaResolution(step, 'fallback_media', 'resolved', manual ? 'Manual image selection accepted.' : 'Selected image accepted.');
      if (step.qa_resolutions?.missing_media) {
        setQaResolution(step, 'missing_media', 'resolved', manual ? 'Manual image attached.' : 'Selected image accepted.');
      }
    }

    function addStepMedia(path, mode = null) {
      const draft = collectSectionDraft();
      if (!draft || !path) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      ensureStepIdentity(step);
      step.media = normalizeMediaList(step.media);
      const replaceIndex = state.sectionMediaPicker.replaceIndex;
      const action = mode || state.sectionMediaPicker.mode || (replaceIndex !== null ? 'replace' : 'append');
      if (action === 'primary' || action === 'hero') {
        step.media = [path, ...step.media.filter(item => item !== path)];
        clearStepMediaFallback(step);
      } else if (replaceIndex !== null && replaceIndex >= 0) {
        step.media[replaceIndex] = path;
        if (replaceIndex === 0) clearStepMediaFallback(step);
      } else if (!step.media.includes(path)) {
        step.media.push(path);
      }
      step.media_settings = normalizeStepMediaSettings(step);
      state.sectionMediaPicker.open = false;
      state.sectionMediaPicker.replaceIndex = null;
      state.sectionMediaPicker.mode = 'append';
      renderSectionEditor();
    }

    function secondsLabel(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) return '';
      const minutes = Math.floor(number / 60);
      const seconds = Math.max(0, number - minutes * 60);
      return `${minutes}:${String(seconds.toFixed(seconds % 1 ? 1 : 0)).padStart(2, '0')}`;
    }

    function stepEvidenceReview(step) {
      step.evidence_review = step.evidence_review || {};
      return step.evidence_review;
    }

    function setEvidenceReview(step, patch = {}) {
      const review = stepEvidenceReview(step);
      Object.assign(review, patch, {updated_at: new Date().toISOString()});
      return review;
    }

    function selectedEvidenceStep() {
      const draft = collectSectionDraft();
      if (!draft) return {};
      const index = selectedSectionStepIndex(draft);
      const step = draft.steps[index];
      ensureStepIdentity(step);
      return {draft, step, index};
    }

    function useEvidenceCandidate(path, mode) {
      if (!path) return;
      const {draft, step} = selectedEvidenceStep();
      if (!draft || !step) return;
      step.media = normalizeMediaList(step.media);
      const primaryMode = mode === 'primary' || mode === 'hero';
      if (primaryMode) {
        step.media = [path, ...step.media.filter(item => item !== path)];
        clearStepMediaFallback(step);
      } else if (!step.media.includes(path)) {
        step.media.push(path);
      }
      step.media_settings = normalizeStepMediaSettings(step);
      setEvidenceReview(step, {
        media_reviewed: true,
        image_manually_selected: true,
        needs_more_media: false,
        note: primaryMode ? 'Primary image manually selected from evidence workbench.' : 'Support image added from evidence workbench.'
      });
      renderSectionEditor();
      toast(primaryMode ? 'Primary image updated.' : 'Support image added.');
    }

    function markEvidenceReviewed() {
      const {draft, step} = selectedEvidenceStep();
      if (!draft || !step) return;
      setEvidenceReview(step, {media_reviewed: true, needs_more_media: false, note: 'Evidence accepted by user.'});
      if (normalizeMediaList(step.media).length) clearStepMediaFallback(step, false);
      renderSectionEditor();
      toast('Evidence accepted.');
    }

    function markEvidenceNeedsMoreMedia() {
      const {draft, step} = selectedEvidenceStep();
      if (!draft || !step) return;
      setEvidenceReview(step, {needs_more_media: true, media_reviewed: false, note: 'User requested more media near this step.'});
      state.sectionMediaPicker.open = true;
      state.sectionMediaPicker.replaceIndex = null;
      state.sectionMediaPicker.mode = 'append';
      state.sectionMediaPicker.filter = 'near';
      renderSectionEditor();
      toast('Showing nearby media options.');
    }

    function nudgeEvidenceTiming(direction) {
      const {draft, step} = selectedEvidenceStep();
      if (!draft || !step) return;
      const delta = direction === 'earlier' ? -2 : 2;
      const target = Number.isFinite(Number(step.target_time)) ? Number(step.target_time) : Number(step.end_time || step.start_time || 0);
      const nextTarget = Math.max(0, target + delta);
      step.target_time = Number(nextTarget.toFixed(2));
      if (Number.isFinite(Number(step.start_time))) step.start_time = Number(Math.max(0, Number(step.start_time) + delta).toFixed(2));
      if (Number.isFinite(Number(step.end_time))) step.end_time = Number(Math.max(step.start_time || 0, Number(step.end_time) + delta).toFixed(2));
      setEvidenceReview(step, {
        timing_adjusted: true,
        media_reviewed: false,
        needs_more_media: true,
        note: `Timing nudged ${direction}; find a better nearby frame.`
      });
      state.sectionMediaPicker.open = true;
      state.sectionMediaPicker.replaceIndex = 0;
      state.sectionMediaPicker.mode = 'replace';
      state.sectionMediaPicker.filter = 'near';
      renderSectionEditor();
      toast(direction === 'earlier' ? 'Timing nudged earlier.' : 'Timing nudged later.');
    }

    async function uploadSectionStepMedia(files) {
      const fileList = Array.from(files || []);
      const draft = collectSectionDraft();
      if (!draft || !state.currentSection || !fileList.length) return;
      const selected = selectedSectionStepIndex(draft);
      const form = new FormData();
      fileList.forEach(file => form.append('files', file));
      try {
        const result = await api(`/api/sections/${state.currentSection}/upload-media`, {method:'POST', body: form});
        const step = draft.steps[selected];
        ensureStepIdentity(step);
        step.media = normalizeMediaList(step.media);
        const existingMediaCount = step.media.length;
        const attachedPaths = [];
        [...(result.added || []), ...(result.skipped || [])].forEach(file => {
          if (file.stored_path && !step.media.includes(file.stored_path)) {
            step.media.push(file.stored_path);
            attachedPaths.push(file.stored_path);
          }
        });
        if (attachedPaths.length && (!existingMediaCount || step.media_fallback)) {
          step.media = [attachedPaths[0], ...step.media.filter(path => path !== attachedPaths[0])];
          clearStepMediaFallback(step);
        }
        step.media_settings = normalizeStepMediaSettings(step);
        const mediaData = await api(`/api/sections/${state.currentSection}/media-options`);
        state.sectionMediaOptions[state.currentSection] = mediaData.media;
        state.sectionDraft = draft;
        state.sectionDraftId = draft.id;
        state.sectionStepIndexes[draft.id] = selected;
        state.sectionMediaPicker.open = false;
        state.sectionMediaPicker.replaceIndex = null;
        state.sectionMediaPicker.mode = 'append';
        renderSectionEditor();
        toast(`${(result.added || []).length} images attached. ${(result.skipped || []).length} skipped.`);
      } catch (err) {
        toast(`Image upload failed: ${err.message}`);
      }
    }

    function removeStepMedia(idx) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      ensureStepIdentity(step);
      step.media = normalizeMediaList(step.media);
      const removed = step.media.splice(idx, 1)[0];
      if (removed && step.media_settings) delete step.media_settings[removed];
      step.media_settings = normalizeStepMediaSettings(step);
      renderSectionEditor();
    }

    function moveStepMedia(idx, dir) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      ensureStepIdentity(step);
      step.media = normalizeMediaList(step.media);
      const next = idx + dir;
      if (next < 0 || next >= step.media.length) return;
      const [item] = step.media.splice(idx, 1);
      step.media.splice(next, 0, item);
      step.media_settings = normalizeStepMediaSettings(step);
      renderSectionEditor();
    }

    function setStepMediaSetting(path, key, value) {
      const draft = collectSectionDraft();
      if (!draft || !path) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      ensureStepIdentity(step);
      if (!step.media.includes(path)) return;
      const current = mediaSettingFor(step, path);
      if (key === 'fit') current.fit = value === 'cover' || value === 'fill' ? 'cover' : 'contain';
      if (key === 'aspect') current.aspect = ['auto', 'portrait', 'landscape', 'square'].includes(value) ? value : 'auto';
      if (key === 'crop_position') current.crop_position = ['center', 'top', 'bottom', 'left', 'right'].includes(value) ? value : 'center';
      step.media_settings[path] = normalizeMediaSetting(current);
      renderSectionEditor();
    }

    function mediaOptionSearchText(option) {
      return [
        option.path,
        option.kind,
        option.source_file,
        option.label,
        option.reason,
        option.summary,
        ...(option.tags || [])
      ].join(' ').toLowerCase();
    }

    function mediaOptionNearStep(option, step) {
      if (option.timestamp_seconds === null || option.timestamp_seconds === undefined) return false;
      const start = Number(step.start_time ?? step.target_time ?? 0);
      const end = Number(step.end_time ?? step.target_time ?? start + 8);
      const ts = Number(option.timestamp_seconds || 0);
      return ts >= Math.max(0, start - 8) && ts <= end + 10;
    }

    function stepMediaCandidateMap(step) {
      return new Map((step.media_candidates || []).map(candidate => [candidate.path, candidate]));
    }

    function filteredSectionMediaOptions(step) {
      const options = state.sectionMediaOptions[state.currentSection] || [];
      const filter = state.sectionMediaPicker.filter || 'near';
      const query = String(state.sectionMediaPicker.search || '').toLowerCase();
      const candidates = stepMediaCandidateMap(step);
      return options.filter(option => {
        const tags = new Set(option.tags || []);
        const tagText = Array.from(tags).join(' ').toLowerCase();
        if (query && !mediaOptionSearchText(option).includes(query)) return false;
        if (filter === 'near') return mediaOptionNearStep(option, step);
        if (filter === 'pointing') return tags.has('pointing') || tagText.includes('pointing');
        if (filter === 'visual_change') return tags.has('visual_change') || tagText.includes('visual_change');
        if (filter === 'sharp') return tags.has('sharp') || tagText.includes('sharp');
        if (filter === 'tool') return tags.has('tool') || tags.has('tool_visible') || tagText.includes('tool');
        if (filter === 'quality') return tags.has('quality') || tags.has('quality_check') || tagText.includes('quality');
        if (filter === 'safety') return tags.has('safety') || tagText.includes('safety');
        return true;
      }).sort((a, b) => {
        const aCandidate = candidates.get(a.path);
        const bCandidate = candidates.get(b.path);
        if (!!aCandidate !== !!bCandidate) return aCandidate ? -1 : 1;
        if (a.is_currently_used !== b.is_currently_used) return a.is_currently_used ? -1 : 1;
        return Number(bCandidate?.score || b.score || 0) - Number(aCandidate?.score || a.score || 0);
      });
    }

    function mediaOptionImage(option) {
      if (option.previewable) {
        return `<img class="media-option-preview clickable" src="${mediaUrl(option.path)}" alt="${esc(option.label || option.path)}" loading="lazy" onclick="openImageZoom('${jsSingleQuoted(option.path)}', '${jsSingleQuoted(option.label || option.path)}')">`;
      }
      return `<div class="source-tile">${esc(option.kind || 'media')}<br><span class="subtle">Preview unavailable</span></div>`;
    }

    function renderEvidenceCandidate(step, candidate) {
      const path = String(candidate.path || '').trim();
      if (!path) return '';
      const used = normalizeMediaList(step.media).includes(path);
      const label = mediaDisplayName(path);
      const tags = candidate.tags || [];
      return `<div class="evidence-candidate ${used ? 'used' : ''}">
        <img src="${mediaUrl(path)}" alt="${esc(label)}" loading="lazy" onclick="openImageZoom('${jsSingleQuoted(path)}', '${jsSingleQuoted(label)}')">
        <div class="evidence-candidate-body">
          <div class="strong media-card-title" title="${esc(path)}">${esc(label)}</div>
          <div class="chip-row">
            ${candidate.timestamp_seconds !== null && candidate.timestamp_seconds !== undefined ? `<span class="chip">~${esc(candidate.timestamp_seconds)}s</span>` : ''}
            ${candidate.score !== null && candidate.score !== undefined ? `<span class="chip">Score ${esc(candidate.score)}</span>` : ''}
            ${candidate.pointing ? '<span class="chip warn">Pointing</span>' : ''}
            ${candidate.after_state ? '<span class="chip good">After-state</span>' : ''}
            ${used ? '<span class="chip good">Used</span>' : ''}
          </div>
          ${tags.length ? `<div class="chip-row">${tags.slice(0, 4).map(tag => `<span class="chip">${esc(tag)}</span>`).join('')}</div>` : ''}
          ${candidate.summary || candidate.reason || candidate.decision ? `<div class="evidence-candidate-note">${esc(candidate.summary || candidate.reason || candidate.decision)}</div>` : ''}
          <div class="media-card-actions">
            <button class="primary" onclick="useEvidenceCandidate('${jsSingleQuoted(path)}', 'primary')">Use as Primary</button>
            <button onclick="useEvidenceCandidate('${jsSingleQuoted(path)}', 'support')">Use as Support</button>
          </div>
        </div>
      </div>`;
    }

    function renderEvidenceWorkbench(draft, step, selected) {
      const review = step.evidence_review || {};
      const candidates = (step.media_candidates || []).filter(item => item.path).slice(0, 12);
      const previous = draft.steps[selected - 1] || null;
      const next = draft.steps[selected + 1] || null;
      const timing = [
        step.start_time !== undefined ? `Start ${secondsLabel(step.start_time)}` : '',
        step.target_time !== undefined ? `Target ${secondsLabel(step.target_time)}` : '',
        step.end_time !== undefined ? `End ${secondsLabel(step.end_time)}` : '',
      ].filter(Boolean).join(' · ') || 'No timing data';
      const neighborTiming = [
        previous ? `Prev ends ${secondsLabel(previous.end_time) || 'unknown'}` : '',
        next ? `Next starts ${secondsLabel(next.start_time) || 'unknown'}` : '',
      ].filter(Boolean).join(' · ') || 'No neighboring timing';
      const transcript = step.raw_transcript || 'No transcript slice captured for this step.';
      return `<div class="evidence-workbench">
        <div class="evidence-head">
          <div>
            <div class="strong">Evidence Workbench</div>
            <div class="subtle">Review timing, transcript, and candidate frames for this step.</div>
          </div>
          <div class="chip-row">
            <span class="chip ${review.media_reviewed ? 'good' : 'warn'}">${review.media_reviewed ? 'Evidence accepted' : 'Needs review'}</span>
            ${review.image_manually_selected ? '<span class="chip good">Manual image</span>' : ''}
            ${review.timing_adjusted ? '<span class="chip warn">Timing adjusted</span>' : ''}
            ${review.needs_more_media ? '<span class="chip warn">Needs media</span>' : ''}
          </div>
        </div>
        <div class="evidence-grid">
          <div class="evidence-panel">
            <div>
              <div class="subtle">Step Timing</div>
              <div class="strong">${esc(timing)}</div>
              <div class="subtle">${esc(neighborTiming)}</div>
            </div>
            <div class="chip-row">
              <button onclick="nudgeEvidenceTiming('earlier')">Too Late</button>
              <button onclick="nudgeEvidenceTiming('later')">Too Early</button>
              <button onclick="markEvidenceNeedsMoreMedia()">Find More Near This Step</button>
              <button class="primary" onclick="markEvidenceReviewed()">Accept Image</button>
            </div>
            <div>
              <div class="subtle">Transcript Slice</div>
              <div class="transcript-slice">${esc(transcript)}</div>
            </div>
            ${review.note ? `<div class="subtle">Review note: ${esc(review.note)}</div>` : ''}
          </div>
          <div class="evidence-panel">
            <div class="card-head">
              <div>
                <div class="strong">Candidates</div>
                <div class="subtle">${candidates.length} scored alternates attached to this step</div>
              </div>
              <span class="chip">${normalizeMediaList(step.media).length} selected</span>
            </div>
            <div class="evidence-candidates">
              ${candidates.map(candidate => renderEvidenceCandidate(step, candidate)).join('') || '<div class="empty compact">No scored candidates for this step. Use Find More Near This Step.</div>'}
            </div>
          </div>
        </div>
      </div>`;
    }

    function renderStepMediaManager(step) {
      const media = normalizeMediaList(step.media);
      const suggestedCount = (step.media_candidates || []).filter(item => item.path).length;
      return `<div class="step-media-manager">
        <div class="step-media-toolbar">
          <div class="chip-row">
            <span class="chip ${media.length ? 'good' : 'warn'}">${esc(media.length)} selected</span>
            ${suggestedCount ? `<span class="chip">${esc(suggestedCount)} alternates</span>` : ''}
            ${step.media_fallback ? '<span class="chip warn">Fallback media</span>' : ''}
            ${step.merge_candidate ? '<span class="chip warn">Merge candidate</span>' : ''}
          </div>
          <div class="split-actions">
            <button onclick="openSectionMediaPicker(null, 'append')">Add Media</button>
            <label class="step-media-upload">Upload Image<input type="file" accept="image/*" multiple onchange="uploadSectionStepMedia(this.files); this.value='';"></label>
            ${media.length ? `<button onclick="openSectionMediaPicker(0, 'replace')">Replace Primary</button>` : ''}
          </div>
        </div>
        ${step.media_fallback_reason ? `<div class="subtle">${esc(step.media_fallback_reason)}</div>` : ''}
        ${step.merge_candidate ? `<div class="question-card pending"><div class="strong">Possible merge</div><div class="subtle">${esc(step.merge_recommendation?.reason || 'This step may be too small to stand alone.')} ${step.merge_recommendation?.direction ? `Consider merging with the ${esc(step.merge_recommendation.direction)} step.` : ''}</div></div>` : ''}
        <div class="selected-step-media">
          ${media.map((path, idx) => mediaCard(step, path, idx, media.length)).join('') || '<div class="empty compact">No media selected for this step.</div>'}
        </div>
        ${renderSectionMediaPicker(step)}
      </div>`;
    }

    function renderSectionMediaPicker(step) {
      if (!state.sectionMediaPicker.open) return '';
      const filters = [
        ['near', 'Near step'],
        ['all', 'All'],
        ['pointing', 'Pointing'],
        ['visual_change', 'Visual change'],
        ['sharp', 'Sharp'],
        ['tool', 'Tool'],
        ['quality', 'Quality'],
        ['safety', 'Safety']
      ];
      const options = filteredSectionMediaOptions(step);
      const candidateMap = stepMediaCandidateMap(step);
      const replacing = state.sectionMediaPicker.replaceIndex !== null;
      return `<div class="media-picker">
        <div class="split-actions">
          <div>
            <div class="strong">${replacing ? 'Replace media' : 'Add media'}</div>
            <div class="subtle" id="media-picker-count">${options.length} options shown</div>
          </div>
          <button onclick="closeSectionMediaPicker()">Close</button>
        </div>
        <div class="chip-row">
          ${filters.map(([value, label]) => `<button class="${state.sectionMediaPicker.filter === value ? 'selected' : ''}" onclick="setSectionMediaPickerFilter('${value}')">${label}</button>`).join('')}
        </div>
        <input class="search" placeholder="Search media" value="${esc(state.sectionMediaPicker.search || '')}" oninput="setSectionMediaPickerSearch(this.value)">
        <div class="media-picker-grid" data-scroll-key="section-media-picker">
          ${options.map(option => `
            <div class="media-option ${option.is_currently_used ? 'used' : ''}" data-media-option data-media-search="${esc(mediaOptionSearchText(option))}">
              ${mediaOptionImage(option)}
              <div class="media-option-title strong">${esc(option.label || option.path)}</div>
              <div class="subtle">${option.timestamp_seconds !== null && option.timestamp_seconds !== undefined ? `~${esc(option.timestamp_seconds)}s · ` : ''}${esc(option.kind || '')}</div>
              <div class="chip-row">
                ${candidateMap.has(option.path) ? `<span class="chip good">Suggested</span>` : ''}
                ${(candidateMap.get(option.path)?.score || option.score) ? `<span class="chip">Score ${esc(candidateMap.get(option.path)?.score || option.score)}</span>` : ''}
                ${option.is_currently_used ? '<span class="chip good">Used</span>' : ''}
                ${(option.tags || []).slice(0, 3).map(tag => `<span class="chip">${esc(tag)}</span>`).join('')}
              </div>
              ${option.summary || option.reason ? `<div class="subtle">${esc(option.summary || option.reason)}</div>` : ''}
              <div class="split-actions">
                ${option.previewable ? `<button onclick="openImageZoom('${jsSingleQuoted(option.path)}', '${jsSingleQuoted(option.label || option.path)}')">Preview</button>` : ''}
                ${replacing ? `<button class="primary" onclick="addStepMedia('${jsSingleQuoted(option.path)}', 'replace')">Use Replacement</button>` : `
                  <button onclick="addStepMedia('${jsSingleQuoted(option.path)}', 'support')">Use as Support</button>
                  <button class="primary" onclick="addStepMedia('${jsSingleQuoted(option.path)}', 'primary')">Use as Primary</button>
                `}
              </div>
            </div>
          `).join('')}<div class="empty" id="media-picker-empty" style="display:${options.length ? 'none' : ''}">No media matches this filter.</div>
        </div>
      </div>`;
    }

    function collectSectionDraft() {
      const draft = state.sectionDraft;
      if (!draft) return null;
      if (!draft.steps || !draft.steps.length) draft.steps = [blankStep()];
      draft.title = document.getElementById('section-title')?.value || draft.title;
      draft.product = document.getElementById('section-product')?.value || '';
      draft.work_area = document.getElementById('section-area')?.value || '';
      draft.station = document.getElementById('section-station')?.value || '';
      draft.status = document.getElementById('section-status')?.value || draft.status;
      draft.tags = commaList(document.getElementById('section-tags')?.value || '');
      const selected = selectedSectionStepIndex(draft);
      if (document.getElementById('step-text')) {
        ['step-tools', 'step-torque', 'step-parts', 'step-safety'].forEach(commitTokenFieldInput);
        const existingStep = draft.steps[selected] || {};
        ensureStepIdentity(existingStep);
        draft.steps[selected] = {
          step_id: existingStep.step_id,
          text: document.getElementById('step-text')?.value || '',
          tools: document.getElementById('step-tools')?.value || '',
          torque: document.getElementById('step-torque')?.value || '',
          quality: document.getElementById('step-quality')?.value || '',
          safety: document.getElementById('step-safety')?.value || '',
          parts: commaList(document.getElementById('step-parts')?.value || ''),
          media: normalizeMediaList(existingStep.media || []),
          media_settings: normalizeStepMediaSettings(existingStep),
          start_time: existingStep.start_time,
          end_time: existingStep.end_time,
          target_time: existingStep.target_time,
          timing_source: existingStep.timing_source,
          timing_confidence: existingStep.timing_confidence,
          timing_word_matches: existingStep.timing_word_matches,
          source_step_id: existingStep.source_step_id,
          source_step_start: existingStep.source_step_start,
          source_step_end: existingStep.source_step_end,
          raw_transcript: existingStep.raw_transcript,
          boundary_reason: existingStep.boundary_reason,
          transcript_confidence: existingStep.transcript_confidence,
          media_candidates: existingStep.media_candidates || [],
          media_match_confidence: existingStep.media_match_confidence,
          media_match_score: existingStep.media_match_score,
          media_fallback: existingStep.media_fallback,
          media_fallback_reason: existingStep.media_fallback_reason,
          merge_candidate: existingStep.merge_candidate,
          merge_recommendation: existingStep.merge_recommendation,
          evidence_review: existingStep.evidence_review || {},
          qa_resolutions: existingStep.qa_resolutions || {}
        };
      }
      ensureDraftStepIds(draft);
      return draft;
    }

    function selectQaIssue(stepIndex, issueId) {
      const draft = collectSectionDraft();
      if (!draft) return;
      state.sectionStepIndexes[draft.id] = Math.min(Math.max(Number(stepIndex) || 0, 0), draft.steps.length - 1);
      state.sectionQaFocus = issueId;
      state.sectionMediaPicker.open = false;
      state.sectionMediaPicker.replaceIndex = null;
      state.sectionMediaPicker.mode = 'append';
      renderSectionEditor();
      requestAnimationFrame(() => focusQaIssue(issueId));
    }

    function focusQaIssue(issueId) {
      const def = qaIssueDefinition(issueId);
      if (def.target === 'media') {
        openSectionMediaPicker(null, 'append');
        return;
      }
      const target = document.getElementById(def.target);
      const tokenEntry = document.getElementById(`${def.target}-entry`);
      if (tokenEntry) {
        tokenEntry.focus();
      } else if (target) {
        target.focus();
        if (target.select) target.select();
      }
    }

    function bypassQaIssue(issueId) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      setQaResolution(step, issueId, 'bypassed', 'User bypassed this QA item.');
      if (issueId === 'merge_candidate') {
        step.merge_candidate = false;
        step.merge_recommendation = null;
      }
      renderSectionEditor();
      toast('QA item bypassed.');
    }

    function resolveQaIssue(issueId, note = 'User resolved this QA item.') {
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      setQaResolution(step, issueId, 'resolved', note);
      renderSectionEditor();
    }

    function acceptStepImage() {
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      if (!normalizeMediaList(step.media).length) {
        toast('Attach an image first.');
        return;
      }
      clearStepMediaFallback(step);
      renderSectionEditor();
      toast('Image accepted.');
    }

    function keepStepSeparate() {
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[selectedSectionStepIndex(draft)];
      step.merge_candidate = false;
      step.merge_recommendation = null;
      setQaResolution(step, 'merge_candidate', 'bypassed', 'User chose to keep this as a separate step.');
      renderSectionEditor();
      toast('Step kept separate.');
    }

    function renderSectionQaPanel(draft, selected) {
      const issues = sectionQaIssues(draft);
      const selectedIssues = issues.filter(issue => issue.stepIndex === selected);
      const handledCount = (draft.steps || []).reduce((total, step) => {
        return total + Object.values(step.qa_resolutions || {}).filter(item => ['bypassed', 'resolved'].includes(item.status)).length;
      }, 0);
      return `<div class="panel section-qa-panel">
        <div class="panel-head">
          <h2>Section QA</h2>
          <span class="chip ${issues.length ? 'warn' : 'good'}">${issues.length ? `${issues.length} open` : 'Clear'}</span>
        </div>
        <div class="panel-body stack">
          ${issues.length ? `
            <div class="qa-list">
              ${issues.slice(0, 12).map(issue => `
                <button class="qa-item ${issue.stepIndex === selected ? 'active' : ''}" onclick="selectQaIssue(${issue.stepIndex}, '${issue.issueId}')">
                  <span class="chip">${issue.stepIndex + 1}</span>
                  <span><strong>${esc(issue.label)}</strong><small>${esc(issue.detail)}</small></span>
                </button>
              `).join('')}
            </div>
          ` : '<div class="empty compact">No open QA items.</div>'}
          <div class="subtle">${selectedIssues.length ? `${selectedIssues.length} on selected step` : 'Selected step clear'}${handledCount ? ` · ${handledCount} handled` : ''}</div>
        </div>
      </div>`;
    }

    function renderSelectedStepQa(draft, step, selected) {
      const issues = stepQaIssues(step, selected);
      if (!issues.length) return '';
      return `<div class="question-card pending step-attention">
        <div class="card-head">
          <div>
            <div class="strong">Needs attention</div>
            <div class="subtle">${issues.length} item${issues.length === 1 ? '' : 's'} on this step</div>
          </div>
          <span class="chip warn">${issues.length}</span>
        </div>
        <div class="qa-list">
          ${issues.map(issue => `
            <div class="qa-attention-row">
              <div>
                <div class="strong">${esc(issue.label)}</div>
                <div class="subtle">${esc(issue.detail)}</div>
              </div>
              <div class="chip-row">
                ${issue.issueId === 'fallback_media' ? '<button onclick="acceptStepImage()">Accept Image</button>' : ''}
                ${issue.issueId === 'merge_candidate' ? `
                  <button onclick="mergeEditorStep(${selected}, -1)" ${selected <= 0 ? 'disabled' : ''}>Merge Previous</button>
                  <button onclick="mergeEditorStep(${selected}, 1)" ${selected >= draft.steps.length - 1 ? 'disabled' : ''}>Merge Next</button>
                  <button onclick="keepStepSeparate()">Keep Separate</button>
                ` : ''}
                ${issue.target && issue.target !== 'merge' ? `<button onclick="focusQaIssue('${issue.issueId}')">Fix</button>` : ''}
                <button onclick="bypassQaIssue('${issue.issueId}')">Bypass</button>
              </div>
            </div>
          `).join('')}
        </div>
      </div>`;
    }

    function renderSectionEditor() {
      const section = state.sections.find(s => s.id === state.currentSection);
      const draft = ensureSectionDraft(section);
      if (!draft) {
        setTitle('Section Editor');
        setTopActions(`<button onclick="go('sections')">Back</button>`);
        document.getElementById('root').innerHTML = `<div class="empty">Section not found.</div>`;
        return;
      }
      setTitle('Section Editor');
      setTopActions(`
        <button onclick="go('sections')">Back</button>
        <button class="danger" onclick="deleteSection('${draft.id}')">Delete</button>
        <button onclick="saveSection()">Save</button>
        <button class="primary" onclick="saveSection('Complete')">Mark Complete</button>
      `);
      const selected = selectedSectionStepIndex(draft);
      const step = draft.steps[selected] || blankStep();
      const qaIssues = sectionQaIssues(draft);
      document.getElementById('root').innerHTML = `
        <div class="toolbar">
          <div>
            <div class="strong">${esc(draft.title)}</div>
            <div class="subtle">${esc(draft.product)} · ${esc(draft.station)} · Source: ${esc(draft.source_job)}</div>
          </div>
          <div class="chip-row"><span class="chip ${draft.status === 'Complete' ? 'good' : 'warn'}">${esc(draft.status)}</span><span class="chip">${draft.steps.length} steps</span><span class="chip ${qaIssues.length ? 'warn' : 'good'}">${qaIssues.length ? `${qaIssues.length} QA` : 'QA clear'}</span></div>
        </div>
        <div class="section-workspace">
          <div class="stack">
            <div class="panel">
              <div class="panel-head"><h2>Section</h2></div>
              <div class="panel-body stack">
                <div class="field"><label>Title</label><input id="section-title" value="${esc(draft.title)}"></div>
                <div class="field"><label>Product</label><input id="section-product" value="${esc(draft.product)}"></div>
                <div class="form-grid">
                  <div class="field"><label>Work Area</label><input id="section-area" value="${esc(draft.work_area)}"></div>
                  <div class="field"><label>Station</label><input id="section-station" value="${esc(draft.station)}"></div>
                </div>
                <div class="form-grid">
                  <div class="field"><label>Status</label><select id="section-status">
                    <option ${draft.status === 'Needs Edit' ? 'selected' : ''}>Needs Edit</option>
                    <option ${draft.status === 'Complete' ? 'selected' : ''}>Complete</option>
                  </select></div>
                  <div class="field"><label>Source Job</label><input value="${esc(draft.source_job || '')}" disabled></div>
                </div>
                <div class="field"><label>Tags</label><input id="section-tags" value="${esc((draft.tags || []).join(', '))}"></div>
              </div>
            </div>
            ${renderSectionQaPanel(draft, selected)}
            <div class="panel">
              <div class="panel-head">
                <h2>Steps</h2>
                <button onclick="addEditorStep()">Add</button>
              </div>
              <div class="panel-body">
                <div class="editor-step-list" data-scroll-key="section-step-list">
                  ${draft.steps.map((item, idx) => `
                    <div class="editor-step-row ${idx === selected ? 'active' : ''}" onclick="selectEditorStep(${idx})">
                      <div class="chip-row">
                        <span class="chip">${idx + 1}</span>
                        ${(item.parts || []).length ? `<span class="chip brand">${(item.parts || []).length} parts</span>` : ''}
                        ${(item.media || []).length ? `<span class="chip">${(item.media || []).length} media</span>` : ''}
                        ${stepQaIssues(item, idx).length ? `<span class="chip warn">${stepQaIssues(item, idx).length} QA</span>` : ''}
                        ${item.media_fallback ? '<span class="chip warn">fallback</span>' : ''}
                        ${item.merge_candidate ? '<span class="chip warn">merge?</span>' : ''}
                      </div>
                      <div class="preview-line strong">${esc(item.text || 'Untitled step')}</div>
                      <div class="preview-line subtle">${esc(item.tools || 'No tools')} · ${esc(item.torque || 'No torque')}</div>
                    </div>
                  `).join('')}
                </div>
              </div>
            </div>
          </div>
          <div class="panel">
            <div class="panel-head">
              <h2>Step ${selected + 1}</h2>
              <div class="chip-row">
                <button class="icon" onclick="moveEditorStep(${selected}, -1)" ${selected <= 0 ? 'disabled' : ''} title="Move up">↑</button>
                <button class="icon" onclick="moveEditorStep(${selected}, 1)" ${selected >= draft.steps.length - 1 ? 'disabled' : ''} title="Move down">↓</button>
                <button onclick="mergeEditorStep(${selected}, -1)" ${selected <= 0 ? 'disabled' : ''}>Merge Previous</button>
                <button onclick="mergeEditorStep(${selected}, 1)" ${selected >= draft.steps.length - 1 ? 'disabled' : ''}>Merge Next</button>
                <button onclick="splitEditorStep(${selected})">Split</button>
                <button class="icon" onclick="duplicateEditorStep(${selected})" title="Duplicate">⧉</button>
                <button class="icon danger" onclick="removeEditorStep(${selected})" title="Delete">×</button>
              </div>
            </div>
            <div class="panel-body stack">
              ${renderSelectedStepQa(draft, step, selected)}
              <div class="field"><label>Instruction</label><textarea id="step-text" data-autogrow data-max-height="420">${esc(step.text)}</textarea></div>
              <div class="form-grid">
                <div class="field"><label>Tools</label>${tokenField('step-tools', step.tools, 'Add tool')}</div>
                <div class="field"><label>Torque</label>${tokenField('step-torque', step.torque, 'Add torque')}</div>
              </div>
              <div class="field"><label>Parts / Consumables</label>${tokenField('step-parts', step.parts || [], 'Add part or consumable')}</div>
              <div class="form-grid">
                <div class="field"><label>Quality Check</label><textarea id="step-quality" data-autogrow data-max-height="260">${esc(step.quality)}</textarea></div>
                <div class="field"><label>Safety</label>${tokenField('step-safety', step.safety, 'Add safety note')}</div>
              </div>
              <div class="field"><label>Step Media</label>${renderStepMediaManager(step)}</div>
              ${renderEvidenceWorkbench(draft, step, selected)}
            </div>
          </div>
        </div>
      `;
    }

    function packageSearchText(section) {
      return sectionSearchText(section);
    }

    function packageListSearchText(packageItem) {
      return [
        packageItem.title,
        packageItem.product,
        packageItem.work_area,
        packageItem.station,
        packageItem.status,
        packageItem.created_at,
        packageItem.updated_at
      ].join(' ').toLowerCase();
    }

    function filterPackageListRows() {
      const query = String(state.packageListSearch || '').toLowerCase();
      let shown = 0;
      document.querySelectorAll('[data-package-list-row]').forEach(row => {
        const match = (row.dataset.packageSearch || '').includes(query);
        row.style.display = match ? '' : 'none';
        if (match) shown += 1;
      });
      const count = document.getElementById('package-list-count');
      if (count) count.textContent = `${shown} packages`;
      const empty = document.getElementById('package-list-empty-row');
      if (empty) empty.style.display = shown ? 'none' : '';
    }

    function filterPackageRows() {
      const query = String(state.packageSearch || '').toLowerCase();
      let shown = 0;
      document.querySelectorAll('[data-package-section-row]').forEach(row => {
        const match = (row.dataset.packageSearch || '').includes(query);
        row.style.display = match ? '' : 'none';
        if (match) shown += 1;
      });
      const count = document.getElementById('package-section-count');
      if (count) count.textContent = `${shown} complete sections`;
      const empty = document.getElementById('package-empty-row');
      if (empty) empty.style.display = shown ? 'none' : '';
    }

    function collectPackageDraftMeta() {
      const packageTitle = document.getElementById('package-title');
      const packageProduct = document.getElementById('package-product');
      const packageArea = document.getElementById('package-area');
      const packageStation = document.getElementById('package-station');
      const packageStatus = document.getElementById('package-status');
      state.packageDraftMeta = {
        title: packageTitle ? packageTitle.value : state.packageDraftMeta.title || '',
        product: packageProduct ? packageProduct.value : state.packageDraftMeta.product || '',
        work_area: packageArea ? packageArea.value : state.packageDraftMeta.work_area || '',
        station: packageStation ? packageStation.value : state.packageDraftMeta.station || '',
        status: packageStatus ? packageStatus.value : state.packageDraftMeta.status || 'Draft'
      };
      return state.packageDraftMeta;
    }

    function draftPackageSections() {
      return state.packageDraft
        .map(id => state.sections.find(section => section.id === id))
        .filter(Boolean);
    }

    function ensurePackageDraft(packageData) {
      if (!packageData || state.packageDraftId === packageData.id) return;
      state.packageDraftId = packageData.id;
      state.packageDraft = [...(packageData.section_ids || [])];
      state.packageDraftMeta = {
        title: packageData.title || '',
        product: packageData.product || '',
        work_area: packageData.work_area || '',
        station: packageData.station || '',
        status: packageData.status || 'Draft'
      };
    }

    function renderPackage() {
      setTitle('Packages');
      setTopActions(`<button class="primary" onclick="createPackageDraft()">New Package</button>`);
      const rows = state.packages || [];
      document.getElementById('root').innerHTML = `
        <div class="toolbar">
          <div class="toolbar-left">
            <input class="search" placeholder="Search packages" value="${esc(state.packageListSearch)}" oninput="state.packageListSearch=this.value; filterPackageListRows()">
          </div>
          <div class="toolbar-right">
            <span class="subtle" id="package-list-count">${rows.length} packages</span>
          </div>
        </div>
        <div class="table-wrap" data-scroll-key="packages-table">
          <table>
            <thead>
              <tr>
                <th>Status</th>
                <th>Title</th>
                <th>Product</th>
                <th>Work Area / Station</th>
                <th>Sections</th>
                <th>Steps</th>
                <th>Updated</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              ${rows.map(packageItem => `
                <tr data-package-list-row data-package-search="${esc(packageListSearchText(packageItem))}">
                  <td>${statusChip(packageItem.status || 'Draft')}</td>
                  <td>
                    <div class="strong">${esc(packageItem.title || 'Instruction Package')}</div>
                    <div class="subtle">Created ${esc(packageItem.created_at || '')}</div>
                  </td>
                  <td>${esc(packageItem.product || '')}</td>
                  <td>${esc(packageItem.work_area || '')}${packageItem.station ? ' / ' + esc(packageItem.station) : ''}</td>
                  <td>${packageItem.section_count || 0}</td>
                  <td>${packageItem.step_count || 0}</td>
                  <td><span class="subtle">${esc(packageItem.updated_at || '')}</span></td>
                  <td>
                    <div class="section-actions">
                      <button onclick="loadPackageDraft('${packageItem.id}')">Edit</button>
                      <button onclick="go('package_view','${packageItem.id}')">Preview</button>
                      <button onclick="go('operator_view','${packageItem.id}')">Operator</button>
                      <button onclick="go('package_print','${packageItem.id}')">Print</button>
                      <button class="danger" onclick="deletePackage('${packageItem.id}')">Delete</button>
                    </div>
                  </td>
                </tr>
              `).join('')}
              <tr id="package-list-empty-row" style="display:none"><td colspan="8"><div class="empty">No matching packages.</div></td></tr>
            </tbody>
          </table>
        </div>
      `;
      filterPackageListRows();
    }

    function renderSectionPackagePreview(section) {
      const steps = (section.steps || []).slice(0, 5);
      const hiddenCount = Math.max(0, (section.steps || []).length - steps.length);
      if (!steps.length) return `<div class="empty compact">No steps in this section.</div>`;
      return `<div class="section-preview-steps">
        ${steps.map((step, idx) => `
          <div class="section-preview-step">
            <div>
              <div class="subtle">Step ${idx + 1}</div>
              <div>${esc(step.text || '')}</div>
            </div>
            <div class="chip-row">
              ${step.tools ? `<span class="chip">Tool: ${esc(step.tools)}</span>` : ''}
              ${step.torque ? `<span class="chip ${step.torque === 'Not required' ? '' : 'warn'}">Torque: ${esc(step.torque)}</span>` : ''}
              ${(step.parts || []).slice(0, 4).map(part => `<span class="chip brand">${esc(part)}</span>`).join('')}
              ${step.quality ? '<span class="chip good">QC</span>' : ''}
              ${step.safety ? '<span class="chip warn">Safety</span>' : ''}
            </div>
            <div class="selected-step-media">${(step.media || []).slice(0, 2).map(m => mediaThumb(m, step)).join('') || '<div class="thumb">No media</div>'}</div>
          </div>
        `).join('')}
        ${hiddenCount ? `<div class="subtle">${hiddenCount} more steps hidden.</div>` : ''}
      </div>`;
    }

    function renderPackageEditor() {
      const packageData = state.currentPackageData;
      if (!packageData) {
        setTitle('Package Editor');
        setTopActions(`<button onclick="go('package')">Back</button>`);
        document.getElementById('root').innerHTML = `<div class="empty">Package not found.</div>`;
        return;
      }
      ensurePackageDraft(packageData);
      const available = state.sections.filter(s => s.status === 'Complete');
      const draftSections = draftPackageSections();
      const meta = state.packageDraftMeta;
      setTitle('Package Editor');
      setTopActions(`
        <button onclick="go('package')">Back</button>
        <button onclick="savePackage()">Save Package</button>
        <button onclick="savePackage('preview')">Preview</button>
        <button class="primary" onclick="savePackage('print')">Print Preview</button>
      `);
      document.getElementById('root').innerHTML = `
        <div class="panel">
          <div class="panel-head">
            <h2>Package Details</h2>
            <span class="chip">${draftSections.length} sections</span>
          </div>
          <div class="panel-body">
            <div class="package-meta-grid">
              <div class="field"><label>Title</label><input id="package-title" value="${esc(meta.title || '')}" placeholder="Instruction package title"></div>
              <div class="field"><label>Product</label><input id="package-product" value="${esc(meta.product || '')}"></div>
              <div class="field"><label>Work Area</label><input id="package-area" value="${esc(meta.work_area || '')}"></div>
              <div class="field"><label>Station</label><input id="package-station" value="${esc(meta.station || '')}"></div>
              <div class="field"><label>Status</label><select id="package-status">
                <option ${meta.status === 'Draft' ? 'selected' : ''}>Draft</option>
                <option ${meta.status === 'Ready for Review' ? 'selected' : ''}>Ready for Review</option>
                <option ${meta.status === 'Published' ? 'selected' : ''}>Published</option>
              </select></div>
            </div>
          </div>
        </div>
        <div style="height:14px"></div>
        <div class="package-editor">
          <div class="panel">
            <div class="panel-head">
              <h2>Complete Sections</h2>
              <span class="chip" id="package-section-count">${available.length} complete sections</span>
            </div>
            <div class="panel-body">
              <div class="toolbar" style="padding:0 0 10px">
                <div class="toolbar-left"><input class="search" placeholder="Search completed sections" value="${esc(state.packageSearch)}" oninput="state.packageSearch=this.value; filterPackageRows()"></div>
              </div>
              <div class="section-result-list" data-scroll-key="package-library">
                ${available.map(s => {
                  const expanded = !!state.expandedPackageSections[s.id];
                  const alreadyAdded = state.packageDraft.includes(s.id);
                  return `
                    <div class="section-result" data-package-section-row data-package-search="${esc(packageSearchText(s))}">
                      <div class="section-result-head">
                        <div>
                          <div class="strong">${esc(s.title)}</div>
                          <div class="subtle">${esc(s.product || '')} · ${esc(s.work_area || '')}${s.station ? ' · ' + esc(s.station) : ''}</div>
                          <div class="chip-row">
                            <span class="chip">${(s.steps || []).length} steps</span>
                            ${(s.tags || []).slice(0, 5).map(tag => `<span class="chip">${esc(tag)}</span>`).join('')}
                          </div>
                        </div>
                        <div class="chip-row">
                          <button onclick="togglePackageSectionPreview('${s.id}')">${expanded ? 'Collapse' : 'Expand'}</button>
                          <button class="primary" onclick="addToPackage('${s.id}')" ${alreadyAdded ? 'disabled' : ''}>${alreadyAdded ? 'Added' : 'Add'}</button>
                        </div>
                      </div>
                      ${expanded ? renderSectionPackagePreview(s) : ''}
                    </div>
                  `;
                }).join('')}
                <div class="empty" id="package-empty-row" style="display:none">No matching complete sections.</div>
              </div>
            </div>
          </div>
          <div class="panel">
            <div class="panel-head">
              <h2>Package Outline</h2>
              <span class="chip">${draftSections.length} sections</span>
            </div>
            <div class="panel-body" data-scroll-key="package-draft">
              <div class="package-outline">
                ${draftSections.map((s, idx) => `
                  <div class="drop-item">
                    <div>
                      <div class="subtle">Section ${idx + 1}</div>
                      <div class="strong">${esc(s.title)}</div>
                      <div class="subtle">${(s.steps || []).length} steps · ${esc(s.source_job || '')}</div>
                      <div class="chip-row">${(s.tags || []).slice(0, 4).map(tag => `<span class="chip">${esc(tag)}</span>`).join('')}</div>
                    </div>
                    <div class="chip-row">
                      <button class="icon" onclick="movePackage(${idx}, -1)" ${idx <= 0 ? 'disabled' : ''}>↑</button>
                      <button class="icon" onclick="movePackage(${idx}, 1)" ${idx >= draftSections.length - 1 ? 'disabled' : ''}>↓</button>
                      <button class="icon danger" onclick="removePackage(${idx})">×</button>
                    </div>
                  </div>
                `).join('') || '<div class="empty">Add complete sections to build this package.</div>'}
              </div>
            </div>
          </div>
        </div>
      `;
      filterPackageRows();
    }

    function renderPackagePreview() {
      const packageData = state.currentPackageData;
      if (!packageData) {
        setTitle('Package Preview');
        setTopActions(`<button onclick="go('package')">Back</button>`);
        document.getElementById('root').innerHTML = `<div class="empty">Package not found.</div>`;
        return;
      }
      const sections = packageData.sections || [];
      const stepCount = sections.reduce((total, section) => total + (section.steps || []).length, 0);
      setTitle('Package Preview');
      setTopActions(`
        <button onclick="go('package')">Back</button>
        <button onclick="loadPackageDraft('${packageData.id}')">Edit Package</button>
        <button onclick="snapshotPackage('${packageData.id}')">Snapshot From Sections</button>
        <button onclick="go('operator_view','${packageData.id}')">Operator View</button>
        <button class="primary" onclick="go('package_print','${packageData.id}')">Print Preview</button>
      `);
      document.getElementById('root').innerHTML = `
        <div class="package-preview">
          <div class="preview-sheet">
            <div class="preview-heading">
              <div>
                <div class="subtle">Instruction Package</div>
                <h2 style="margin:4px 0 0">${esc(packageData.title)}</h2>
              </div>
              <div class="chip-row">
                ${statusChip(packageData.status || 'Draft')}
                <span class="chip">${sections.length} sections</span>
                <span class="chip">${stepCount} steps</span>
              </div>
            </div>
            <div class="form-grid">
              <div><div class="subtle">Product</div><div class="strong">${esc(packageData.product || '')}</div></div>
              <div><div class="subtle">Work Area / Station</div><div class="strong">${esc(packageData.work_area || '')} ${packageData.station ? '· ' + esc(packageData.station) : ''}</div></div>
              <div><div class="subtle">Created</div><div class="strong">${esc(packageData.created_at || '')}</div></div>
              <div><div class="subtle">Updated</div><div class="strong">${esc(packageData.updated_at || '')}</div></div>
            </div>
            ${sections.map((section, sectionIndex) => `
              <div class="preview-section">
                <div>
                  <div class="subtle">Section ${sectionIndex + 1}</div>
                  <h3 style="margin:2px 0 4px">${esc(section.title)}</h3>
                  <div class="subtle">${esc(section.product || '')} · ${esc(section.work_area || '')} · ${esc(section.station || '')} · Source: ${esc(section.source_job || '')}</div>
                  <div class="chip-row">${(section.tags || []).map(tag => `<span class="chip">${esc(tag)}</span>`).join('')}</div>
                </div>
                ${(section.steps || []).map((step, stepIndex) => `
                  <div class="preview-step">
                    <div class="subtle">Step ${stepIndex + 1}</div>
                    <div class="preview-step-text">${esc(step.text || '')}</div>
                    <div class="chip-row">
                      ${step.tools ? `<span class="chip">Tool: ${esc(step.tools)}</span>` : ''}
                      ${step.torque ? `<span class="chip ${step.torque === 'Not required' ? '' : 'warn'}">Torque: ${esc(step.torque)}</span>` : ''}
                      ${step.quality ? `<span class="chip good">QC</span>` : ''}
                      ${step.safety ? `<span class="chip warn">Safety</span>` : ''}
                    </div>
                    ${(step.parts || []).length ? `<div class="chip-row">${(step.parts || []).map(part => `<span class="chip brand">${esc(part)}</span>`).join('')}</div>` : ''}
                    ${step.quality ? `<div><div class="subtle">Quality Check</div><div>${esc(step.quality)}</div></div>` : ''}
                    ${step.safety ? `<div><div class="subtle">Safety</div><div>${esc(step.safety)}</div></div>` : ''}
                    ${(step.media || []).length ? `
                      <div>
                        <div class="subtle">Media</div>
                        <div class="selected-step-media">${(step.media || []).map(path => mediaThumb(path, step)).join('')}</div>
                        <div class="media-reference-list">${(step.media || []).map(path => `<div>${esc(path)}</div>`).join('')}</div>
                      </div>` : ''}
                  </div>
                `).join('')}
              </div>
            `).join('') || '<div class="empty">No sections saved in this package.</div>'}
          </div>
        </div>
      `;
    }

    function printablePackageSteps(packageData) {
      const items = [];
      (packageData.sections || []).forEach((section, sectionIndex) => {
        (section.steps || []).forEach((step, stepIndex) => {
          items.push({
            globalStep: items.length + 1,
            sectionIndex,
            stepIndex,
            sectionTitle: section.title || `Section ${sectionIndex + 1}`,
            sectionMeta: [section.product, section.work_area, section.station].filter(Boolean).join(' · '),
            step
          });
        });
      });
      return items;
    }

    function estimatePrintStepWeight(item) {
      const step = item.step || {};
      const textLength = String(step.text || '').length;
      const mediaCount = (step.media || []).filter(Boolean).length;
      const partsCount = (step.parts || []).length;
      let weight = 1;
      if (textLength > 160) weight += .7;
      else if (textLength > 105) weight += .35;
      if (mediaCount > 1) weight += .55;
      if (step.quality) weight += .45;
      if (step.safety) weight += .45;
      if (partsCount > 5) weight += .35;
      if (`${step.tools || ''} ${step.torque || ''}`.length > 70) weight += .25;
      return weight;
    }

    function canUseThreePrintSteps(items) {
      if (items.length !== 3) return false;
      const weights = items.map(estimatePrintStepWeight);
      const hasHeavyDetail = items.some(item => {
        const step = item.step || {};
        return String(step.text || '').length > 120
          || (step.media || []).filter(Boolean).length > 1
          || !!step.quality
          || !!step.safety
          || (step.parts || []).length > 5;
      });
      return !hasHeavyDetail && weights.every(weight => weight <= 1.45) && weights.reduce((sum, weight) => sum + weight, 0) <= 3.75;
    }

    function shouldUseSinglePrintStep(item) {
      const step = item.step || {};
      const weight = estimatePrintStepWeight(item);
      return weight >= 2.15
        || String(step.text || '').length > 230
        || ((step.media || []).filter(Boolean).length > 1 && (!!step.quality || !!step.safety || (step.parts || []).length > 6));
    }

    function chunkPrintableSteps(items) {
      const pages = [];
      let index = 0;
      while (index < items.length) {
        const remaining = items.length - index;
        let count = 1;
        if (remaining >= 3 && canUseThreePrintSteps(items.slice(index, index + 3))) {
          count = 3;
        } else if (remaining >= 2 && !shouldUseSinglePrintStep(items[index])) {
          count = 2;
        }
        pages.push(items.slice(index, index + count));
        index += count;
      }
      return pages;
    }

    function printMediaSlot(path, role, step = null) {
      const value = String(path || '').trim();
      const label = role === 'support' ? 'Support image' : 'Primary image';
      const fallback = role === 'support' ? 'Support image unavailable' : 'No image selected';
      const setting = mediaSettingFor(step || {}, value);
      if (!value || !(value.match(/\.(jpg|jpeg|png|webp)$/i) || value.startsWith('storage/'))) {
        return `<div class="print-media-slot ${role} missing"><div class="print-media-fallback">${fallback}</div></div>`;
      }
      return `<div class="print-media-slot ${role} ${mediaClassNames(setting)}">
        <img src="${mediaUrl(value)}" alt="${esc(label)}" loading="lazy" onerror="this.parentElement.classList.add('missing'); this.remove();">
        <div class="print-media-fallback">${role === 'support' ? 'Support image unavailable' : 'Image unavailable'}</div>
      </div>`;
    }

    function renderPrintMedia(step) {
      const media = (step.media || []).filter(Boolean).slice(0, 2);
      if (!media.length) {
        return `<div class="print-media single">${printMediaSlot('', 'primary', step)}</div>`;
      }
      return `<div class="print-media ${media.length === 1 ? 'single' : ''}">
        ${printMediaSlot(media[0], 'primary', step)}
        ${media[1] ? printMediaSlot(media[1], 'support', step) : ''}
      </div>`;
    }

    function renderPrintCallout(title, content, cls = '') {
      return `<div class="print-callout ${cls}">
        <h4>${esc(title)}</h4>
        <div>${content}</div>
      </div>`;
    }

    function printPills(values, emptyText = 'None listed', cls = '') {
      const items = tokenValues(values);
      if (!items.length) return `<span class="print-muted">${esc(emptyText)}</span>`;
      return `<div class="print-pill-list">${items.map(item => `<span class="print-pill ${cls}">${esc(item)}</span>`).join('')}</div>`;
    }

    function renderPrintStep(item) {
      const step = item.step || {};
      const parts = printPills(step.parts || [], 'None listed');
      const torque = step.torque || 'Not required';
      const torqueRequired = torque && torque !== 'Not required';
      const tools = step.tools || 'None listed';
      const toolsTorque = `${printPills(tools, 'None listed')}<div style="margin-top:4px">${printPills(torque, 'Not required', torqueRequired ? 'warn' : 'muted')}</div>`;
      const callouts = [
        renderPrintCallout('Parts / Consumables', parts),
        renderPrintCallout('Tools & Torque', toolsTorque, torqueRequired ? 'torque-required' : ''),
        step.quality ? renderPrintCallout('Quality', esc(step.quality), 'quality') : '',
        step.safety ? renderPrintCallout('Safety', printPills(step.safety, '', 'warn'), 'safety') : ''
      ].filter(Boolean).join('');
      return `<article class="print-step-card">
        <div class="print-step-kicker">
          <span>Step ${item.globalStep}</span>
          <span>${esc(item.sectionTitle)}</span>
        </div>
        <div class="print-step-text">${esc(step.text || 'No instruction text.')}</div>
        ${renderPrintMedia(step)}
        <div class="print-callout-grid">${callouts}</div>
      </article>`;
    }

    function renderPackagePrint() {
      const packageData = state.currentPackageData;
      if (!packageData) {
        setTitle('Print Preview');
        setTopActions(`<button onclick="go('package')">Back</button>`);
        document.getElementById('root').innerHTML = `<div class="empty">Package not found.</div>`;
        return;
      }
      const steps = printablePackageSteps(packageData);
      const pages = chunkPrintableSteps(steps);
      setTitle('Print Preview');
      setTopActions(`
        <button onclick="go('package_view','${packageData.id}')">Back</button>
        <button class="primary" onclick="window.print()">Print</button>
      `);
      if (!steps.length) {
        document.getElementById('root').innerHTML = `
          <div class="print-preview">
            <section class="print-page">
              <div class="print-page-head">
                <div>
                  <div class="subtle">Instruction Package</div>
                  <div class="print-page-title">${esc(packageData.title || 'Untitled Package')}</div>
                </div>
                <div class="print-page-meta"><span>No steps</span>${statusChip(packageData.status || 'Draft')}</div>
              </div>
              <div class="empty">No sections or steps saved in this package.</div>
              <div class="print-page-foot"><span>${esc(packageData.product || '')}</span><span>Page 1 of 1</span></div>
            </section>
          </div>
        `;
        return;
      }
      const pageTotal = pages.length;
      document.getElementById('root').innerHTML = `
        <div class="print-preview">
          ${pages.map((pageSteps, pageIndex) => `
            <section class="print-page">
              <div class="print-page-head">
                <div>
                  <div class="subtle">Instruction Package</div>
                  <div class="print-page-title">${esc(packageData.title || 'Untitled Package')}</div>
                  <div class="subtle">${esc(packageData.product || '')}${packageData.work_area ? ' · ' + esc(packageData.work_area) : ''}${packageData.station ? ' · ' + esc(packageData.station) : ''}</div>
                </div>
                <div class="print-page-meta">
                  ${statusChip(packageData.status || 'Draft')}
                  <span>${steps.length} steps</span>
                  <span>Page ${pageIndex + 1} of ${pageTotal}</span>
                </div>
              </div>
              <div class="print-steps steps-${pageSteps.length}">
                ${pageSteps.map(renderPrintStep).join('')}
              </div>
              <div class="print-page-foot">
                <span>Snapshot updated ${esc(packageData.updated_at || '')}</span>
                <span>Page ${pageIndex + 1} of ${pageTotal}</span>
              </div>
            </section>
          `).join('')}
        </div>
      `;
    }

    function setOperatorStep(packageId, index) {
      const packageData = state.currentPackageData;
      const steps = printablePackageSteps(packageData || {});
      if (!steps.length) return;
      state.operatorStepIndexes[packageId] = Math.max(0, Math.min(steps.length - 1, Number(index) || 0));
      renderOperatorView();
    }

    function moveOperatorStep(packageId, delta) {
      const current = state.operatorStepIndexes[packageId] || 0;
      setOperatorStep(packageId, current + delta);
    }

    function renderOperatorCallout(title, content, cls = '') {
      if (!content) return '';
      return `<div class="operator-callout ${cls}">
        <div class="subtle">${esc(title)}</div>
        <div>${content}</div>
      </div>`;
    }

    function renderOperatorView() {
      const packageData = state.currentPackageData;
      if (!packageData) {
        setTitle('Operator View');
        setTopActions(`<button onclick="go('package')">Back</button>`);
        document.getElementById('root').innerHTML = `<div class="empty">Package not found.</div>`;
        return;
      }
      const steps = printablePackageSteps(packageData);
      const index = Math.max(0, Math.min(steps.length - 1, state.operatorStepIndexes[packageData.id] || 0));
      state.operatorStepIndexes[packageData.id] = index;
      setTitle('Operator View');
      setTopActions(`
        <button onclick="go('package_view','${packageData.id}')">Back</button>
        <button onclick="toggleNav()">Toggle Nav</button>
        <button onclick="go('package_print','${packageData.id}')">Print</button>
      `);
      if (!steps.length) {
        document.getElementById('root').innerHTML = `<div class="empty">No steps saved in this package.</div>`;
        return;
      }
      const item = steps[index];
      const step = item.step || {};
      const media = normalizeMediaList(step.media);
      const primaryImage = media[0] || '';
      const parts = printPills(step.parts || [], 'None listed');
      const tools = printPills(step.tools || '', 'None listed');
      const torque = printPills(step.torque || 'Not required', 'Not required', step.torque && step.torque !== 'Not required' ? 'warn' : 'muted');
      document.getElementById('root').innerHTML = `
        <div class="operator-shell">
          <div class="toolbar">
            <div>
              <div class="strong">${esc(packageData.title || 'Instruction Package')}</div>
              <div class="subtle">${esc(packageData.product || '')}${packageData.work_area ? ' · ' + esc(packageData.work_area) : ''}${packageData.station ? ' · ' + esc(packageData.station) : ''}</div>
            </div>
            <div class="chip-row">
              ${statusChip(packageData.status || 'Draft')}
              <span class="chip">${index + 1}/${steps.length}</span>
              <span class="chip">${esc(item.sectionTitle)}</span>
            </div>
          </div>
          <div class="operator-stage">
            <div class="operator-media">
              ${primaryImage ? `<img src="${mediaUrl(primaryImage)}" alt="${esc(item.sectionTitle)} step ${index + 1}" onclick="openImageZoom('${jsSingleQuoted(primaryImage)}', '${jsSingleQuoted(item.sectionTitle)} step ${index + 1}')">` : '<div class="empty compact">No image selected for this step.</div>'}
            </div>
            <div class="operator-step-panel">
              <div>
                <div class="subtle">Step ${item.globalStep} · ${esc(item.sectionTitle)}</div>
                <div class="operator-step-text">${esc(step.text || 'No instruction text.')}</div>
              </div>
              <div class="chip-row">
                ${(step.media || []).length ? `<span class="chip good">${(step.media || []).length} media</span>` : '<span class="chip warn">No media</span>'}
                ${step.media_match_confidence ? `<span class="chip">${esc(step.media_match_confidence)}</span>` : ''}
                ${step.timing_confidence ? `<span class="chip">Timing ${esc(step.timing_confidence)}</span>` : ''}
              </div>
              <div class="operator-callouts">
                ${renderOperatorCallout('Parts / Consumables', parts)}
                ${renderOperatorCallout('Tools', tools)}
                ${renderOperatorCallout('Torque', torque, step.torque && step.torque !== 'Not required' ? 'safety' : '')}
                ${step.quality ? renderOperatorCallout('Quality Check', esc(step.quality), 'quality') : ''}
                ${step.safety ? renderOperatorCallout('Safety', printPills(step.safety, '', 'warn'), 'safety') : ''}
              </div>
            </div>
          </div>
          <div class="operator-nav">
            <button onclick="moveOperatorStep('${packageData.id}', -1)" ${index <= 0 ? 'disabled' : ''}>Previous</button>
            <div class="operator-progress">
              <div class="bar"><span style="width:${Math.round(((index + 1) / steps.length) * 100)}%"></span></div>
              <div>Step ${index + 1} of ${steps.length}</div>
            </div>
            <button class="primary" onclick="moveOperatorStep('${packageData.id}', 1)" ${index >= steps.length - 1 ? 'disabled' : ''}>Next</button>
          </div>
        </div>
      `;
    }

    function renderSettings() {
      setTitle('Settings');
      setTopActions(`<button class="primary" onclick="saveSettings()">Save</button>`);
      const settings = state.settings || {};
      document.getElementById('root').innerHTML = `
        <div class="panel" style="max-width:760px">
          <div class="panel-head"><h2>Settings</h2></div>
          <div class="panel-body stack">
            <div class="field">
              <label>OpenAI API key</label>
              <input id="settings-openai-key" type="password" placeholder="${settings.has_openai_api_key ? 'Key saved. Leave blank to keep it.' : 'Paste API key'}">
            </div>
            <div class="form-grid">
              <div class="field"><label>Transcription backend</label><select id="settings-transcription-backend">
                <option value="auto" ${settings.transcription_backend === 'auto' ? 'selected' : ''}>Auto - local if configured</option>
                <option value="local_whisper" ${settings.transcription_backend === 'local_whisper' ? 'selected' : ''}>Local whisper.cpp</option>
                <option value="openai" ${settings.transcription_backend === 'openai' ? 'selected' : ''}>OpenAI API</option>
              </select></div>
              <div class="field"><label>Transcription model</label><select id="settings-transcription-model">
                <option ${settings.transcription_model === 'gpt-4o-mini-transcribe' ? 'selected' : ''}>gpt-4o-mini-transcribe</option>
                <option ${settings.transcription_model === 'gpt-4o-transcribe' ? 'selected' : ''}>gpt-4o-transcribe</option>
                <option ${settings.transcription_model === 'whisper-1' ? 'selected' : ''}>whisper-1</option>
              </select></div>
              <div class="field"><label>Local whisper CLI</label><input id="settings-local-whisper-cli" value="${esc(settings.local_whisper_cli_path || '')}" placeholder="/opt/homebrew/bin/whisper-cli"></div>
              <div class="field"><label>Local whisper model</label><input id="settings-local-whisper-model" value="${esc(settings.local_whisper_model_path || '')}" placeholder="/path/to/ggml-small.en.bin"></div>
              <div class="field"><label>Default processing</label><select id="settings-default-processing">
                <option ${settings.default_processing === 'Fast' ? 'selected' : ''}>Fast</option>
                <option ${settings.default_processing === 'Standard' ? 'selected' : ''}>Standard</option>
                <option ${settings.default_processing === 'Deep' ? 'selected' : ''}>Deep</option>
              </select></div>
            </div>
            <div class="chip-row">
              <span class="chip ${settings.has_local_whisper_cli ? 'good' : 'warn'}">CLI: ${settings.has_local_whisper_cli ? 'found' : 'missing'}</span>
              <span class="chip ${settings.has_local_whisper_model ? 'good' : 'warn'}">Local model: ${settings.has_local_whisper_model ? 'found' : 'missing'}</span>
              <span class="chip">Effective: ${esc(settings.effective_transcription_backend || 'openai')}</span>
            </div>
            <label class="chip-row" style="justify-content:flex-start">
              <input id="settings-local-whisper-no-gpu" type="checkbox" style="width:auto; min-height:auto" ${settings.local_whisper_no_gpu ? 'checked' : ''}>
              <span>Run local Whisper CPU-only</span>
            </label>
            <label class="chip-row" style="justify-content:flex-start">
              <input id="settings-transcription-enabled" type="checkbox" style="width:auto; min-height:auto" ${settings.transcription_enabled ? 'checked' : ''}>
              <span>Run transcription when processing jobs</span>
            </label>
            <div class="field"><label>Storage root</label><input value="${esc(settings.storage_root || 'storage')}" disabled></div>
            <div class="split-actions" style="justify-content:flex-start">
              <button onclick="exportProcessingLog()">Export Processing Log</button>
            </div>
            <div class="field">
              <label>Operator phrases</label>
              <div class="stage-detail-list">
                ${(settings.operator_phrase_guide || []).map(group => `
                  <div class="file-row">
                    <div>
                      <div class="strong">${esc(group.label)}</div>
                      <div class="subtle">${esc(group.purpose)}</div>
                      <div class="chip-row" style="margin-top:6px">${(group.phrases || []).map(phrase => `<span class="chip">${esc(phrase)}</span>`).join('')}</div>
                    </div>
                  </div>
                `).join('')}
              </div>
            </div>
          </div>
        </div>
      `;
    }

    function render(refreshData = true) {
      saveScrollPositions();
      readHash();
      if (refreshData) {
        loadData().then(() => render(false)).catch(err => toast(err.message));
        return;
      }
      const fingerprint = pageFingerprint();
      if (fingerprint === state.renderFingerprint) {
        return;
      }
      state.renderFingerprint = fingerprint;
      if (state.currentPage === 'jobs') renderJobs();
      if (state.currentPage === 'job') renderJob();
      if (state.currentPage === 'sections') renderSections();
      if (state.currentPage === 'section') renderSectionEditor();
      if (state.currentPage === 'package') renderPackage();
      if (state.currentPage === 'package_edit') renderPackageEditor();
      if (state.currentPage === 'package_view') renderPackagePreview();
      if (state.currentPage === 'package_print') renderPackagePrint();
      if (state.currentPage === 'operator_view') renderOperatorView();
      if (state.currentPage === 'settings') renderSettings();
      restoreScrollPositions();
    }

    function openJobModal() {
      document.getElementById('jobModal').classList.add('open');
    }

    function closeJobModal() {
      document.getElementById('jobModal').classList.remove('open');
    }

    async function createJob() {
      const input = document.getElementById('newFiles');
      const files = Array.from(input.files || []);
      if (!files.length) {
        toast('Choose at least one file before starting a job.');
        return;
      }
      const payload = {
        name: document.getElementById('newName').value,
        product: document.getElementById('newProduct').value,
        work_area: document.getElementById('newArea').value,
        station: document.getElementById('newStation').value,
        profile: document.getElementById('newProfile').value,
        expected_step_count: document.getElementById('newExpectedSteps').value || null,
        focus_area: document.getElementById('newFocusArea').value || 'center',
        files: []
      };
      try {
        const res = await api('/api/jobs', {method: 'POST', body: JSON.stringify(payload)});
        const uploadResult = await uploadJobFiles(res.job.id, files);
        const uploadedCount = (uploadResult.added || []).length + (uploadResult.skipped || []).length;
        if (!uploadedCount) throw new Error('No files were accepted by the upload parser.');
        closeJobModal();
        await api(`/api/jobs/${res.job.id}/start`, {method: 'POST', body: '{}'});
        go('job', res.job.id);
      } catch (err) {
        toast(`Job upload failed: ${err.message}`);
        await loadData();
        render(false);
      }
    }

    async function saveSettings() {
      const payload = {
        openai_api_key: document.getElementById('settings-openai-key')?.value || '',
        transcription_backend: document.getElementById('settings-transcription-backend')?.value || 'auto',
        transcription_model: document.getElementById('settings-transcription-model')?.value || 'whisper-1',
        local_whisper_cli_path: document.getElementById('settings-local-whisper-cli')?.value || '',
        local_whisper_model_path: document.getElementById('settings-local-whisper-model')?.value || '',
        local_whisper_no_gpu: !!document.getElementById('settings-local-whisper-no-gpu')?.checked,
        transcription_enabled: !!document.getElementById('settings-transcription-enabled')?.checked,
        default_processing: document.getElementById('settings-default-processing')?.value || 'Standard'
      };
      const result = await api('/api/settings', {method: 'POST', body: JSON.stringify(payload)});
      state.settings = result.settings;
      toast('Settings saved.');
      render(false);
    }

    async function exportProcessingLog() {
      const res = await fetch('/api/export/processing-log');
      if (!res.ok) {
        toast('Export failed.');
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      const disposition = res.headers.get('Content-Disposition') || '';
      const match = /filename="([^"]+)"/.exec(disposition);
      link.href = url;
      link.download = match ? match[1] : 'assembly_assistant_processing_log.json';
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      try {
        const clearRes = await fetch('/api/export/processing-log/clear', {method: 'POST', body: '{}'});
        if (clearRes.ok) {
          await loadData();
          invalidateRender();
          render(false);
          toast('Processing log exported and cleared.');
          return;
        }
      } catch (err) {
        console.warn('Processing log clear failed', err);
      }
      toast('Processing log exported. Clear failed.');
    }

    async function uploadLargeJobFile(id, file) {
      const chunkSize = 16 * 1024 * 1024;
      const uploadId = `${Date.now()}-${Math.random().toString(16).slice(2)}-${file.name}`.replace(/[^A-Za-z0-9._-]+/g, '_');
      let offset = 0;
      let finalResult = {added: [], skipped: []};
      while (offset < file.size) {
        const end = Math.min(file.size, offset + chunkSize);
        const params = new URLSearchParams({
          upload_id: uploadId,
          name: file.name || 'upload.bin',
          type: file.type || 'application/octet-stream',
          size: String(file.size || 0),
          offset: String(offset),
          final: end >= file.size ? '1' : '0'
        });
        const res = await fetch(`/api/jobs/${id}/upload-chunk?${params.toString()}`, {
          method: 'POST',
          headers: {'Content-Type': 'application/octet-stream'},
          body: file.slice(offset, end)
        });
        if (!res.ok) throw new Error(await res.text());
        const result = await res.json();
        if (result.complete) finalResult = result;
        offset = end;
      }
      return finalResult;
    }

    async function uploadJobFiles(id, files) {
      const fileList = Array.from(files || []);
      if (!fileList.length) return {added: [], skipped: []};
      const largeThreshold = 100 * 1024 * 1024;
      const smallFiles = fileList.filter(file => (file.size || 0) <= largeThreshold);
      const largeFiles = fileList.filter(file => (file.size || 0) > largeThreshold);
      const added = [];
      const skipped = [];
      if (largeFiles.length) toast(`Uploading ${largeFiles.length} large file${largeFiles.length === 1 ? '' : 's'} in chunks...`);
      if (smallFiles.length) {
        const form = new FormData();
        smallFiles.forEach(file => form.append('files', file));
        const result = await api(`/api/jobs/${id}/upload`, {method: 'POST', body: form});
        added.push(...(result.added || []));
        skipped.push(...(result.skipped || []));
      }
      for (const file of largeFiles) {
        const result = await uploadLargeJobFile(id, file);
        added.push(...(result.added || []));
        skipped.push(...(result.skipped || []));
      }
      return {added, skipped};
    }

    async function uploadQuestionFiles(id, files) {
      const fileList = Array.from(files || []);
      if (!fileList.length) return {added: [], skipped: []};
      const form = new FormData();
      fileList.forEach(file => form.append('files', file));
      return api(`/api/questions/${id}/upload`, {method: 'POST', body: form});
    }

    async function startJob(id) {
      let result = null;
      try {
        result = await api(`/api/jobs/${id}/start`, {method: 'POST', body: '{}'});
      } catch (err) {
        toast(`Processing did not start: ${err.message}`);
        await loadData();
        render();
        return;
      }
      if (!result.ok) {
        toast(result.error || 'Use Advanced to rerun this job.');
        render();
        return;
      }
      toast('Processing started.');
      render();
    }

    async function rerunJob(id) {
      try {
        await api(`/api/jobs/${id}/rerun`, {method: 'POST', body: '{}'});
        toast('Rerun started.');
        render();
      } catch (err) {
        toast(`Rerun did not start: ${err.message}`);
        await loadData();
        render();
      }
    }

    async function rerunStage(id, stage) {
      const result = await api(`/api/jobs/${id}/rerun-stage`, {
        method: 'POST',
        body: JSON.stringify({stage})
      });
      if (!result.ok) {
        toast('Stage rerun is not available right now.');
        render();
        return;
      }
      toast('Stage rerun started.');
      render();
    }

    async function resumeCurrentStage(id) {
      const job = state.currentJobData || state.jobs.find(item => item.id === id);
      const stage = job?.watchdog?.stage
        || stageKeys.find(key => job?.stages?.[key]?.status === 'failed')
        || stageKeys.find(key => job?.stages?.[key]?.status === 'running')
        || activeStageKey(job || {});
      const supported = {
        transcription: 'transcription',
        drawings: 'drawings',
        video: 'video',
        draft_sections: 'draft_sections'
      };
      if (supported[stage]) {
        await rerunStage(id, supported[stage]);
        return;
      }
      if (job?.draft_sections?.length) {
        await rerunStage(id, 'media_assignment');
        return;
      }
      await rerunJob(id);
    }

    async function completeJob(id) {
      await api(`/api/jobs/${id}/complete`, {method: 'POST', body: '{}'});
      toast('Sections moved to library.');
      go('sections');
    }

    async function deleteJob(id) {
      if (!confirm('Delete this job from the active job list? Stored media will be retained.')) return;
      await api(`/api/jobs/${id}/delete`, {method: 'POST', body: '{}'});
      toast('Job deleted.');
      go('jobs');
    }

    async function removeUpload(jobId, identity) {
      if (!confirm('Remove this upload from the processing set? Stored media will be retained.')) return;
      await api(`/api/jobs/${jobId}/remove-file`, {
        method: 'POST',
        body: JSON.stringify({identity})
      });
      toast('Upload removed. Run processing again when ready.');
      await loadData();
      invalidateRender();
      render(false);
    }

    function toggleSourceSyncFields(domId) {
      const manual = document.getElementById(`source-sync-mode-${domId}`)?.value === 'manual';
      document.querySelectorAll(`.source-manual-${domId}`).forEach(element => element.classList.toggle('hidden', !manual));
    }

    async function saveSourceSetup(jobId, sourceId, domId) {
      const value = id => document.getElementById(id)?.value ?? '';
      const checked = id => !!document.getElementById(id)?.checked;
      const payload = {
        identity: sourceId,
        source_role: value(`source-role-${domId}`),
        source_label: value(`source-label-${domId}`),
        authority_level: value(`source-authority-${domId}`),
        revision: value(`source-revision-${domId}`),
        applicability: value(`source-applicability-${domId}`),
        use_for_drafting: checked(`source-narration-${domId}`),
        timeline_reference: checked(`source-reference-${domId}`),
        sync_mode: value(`source-sync-mode-${domId}`),
        start_source_seconds: value(`source-start-source-${domId}`),
        start_timeline_seconds: value(`source-start-timeline-${domId}`),
        end_source_seconds: value(`source-end-source-${domId}`),
        end_timeline_seconds: value(`source-end-timeline-${domId}`)
      };
      try {
        await api(`/api/jobs/${jobId}/source`, {method: 'POST', body: JSON.stringify(payload)});
        toast('Source setup saved. Processing must be run again to apply it.');
        await loadData();
        invalidateRender();
        render(false);
      } catch (err) {
        toast(`Source setup was not saved: ${err.message}`);
      }
    }

    function openAddFiles(id) {
      const input = document.createElement('input');
      input.type = 'file';
      input.multiple = true;
      input.onchange = async () => {
        try {
          const result = await uploadJobFiles(id, input.files);
          toast(`${result.added.length} files added. ${result.skipped.length} duplicates skipped.`);
          await loadData();
          render();
        } catch (err) {
          toast(`Upload failed: ${err.message}`);
          await loadData();
          render();
        }
      };
      input.click();
    }

    function selectQuestionChoice(id, choice) {
      state.selectedChoices[id] = choice;
      state.questionDrafts[id] = choice;
      renderJob();
    }

    function captureQuestionFiles(id, fileList) {
      const files = Array.from(fileList || []);
      state.questionFileObjects[id] = files;
      state.questionFiles[id] = files.map(file => ({
        name: file.name,
        type: file.type || 'unknown',
        size: `${Math.round(file.size / 1024)} KB`
      }));
      renderJob();
    }

    async function answerQuestion(id, status) {
      const typed = document.getElementById(`answer-${id}`)?.value || '';
      const selected = state.selectedChoices[id] || '';
      const attachment = document.getElementById(`attach-${id}`)?.value || '';
      const uploadResult = await uploadQuestionFiles(id, state.questionFileObjects[id] || []);
      const uploadedFiles = [...(uploadResult.added || []), ...(uploadResult.skipped || [])];
      const attachmentFiles = uploadedFiles.length ? uploadedFiles : (state.questionFiles[id] || []);
      await api(`/api/questions/${id}/answer`, {
        method: 'POST',
        body: JSON.stringify({status, answer: typed || selected, attachment_note: attachment, attachment_files: attachmentFiles})
      });
      delete state.questionDrafts[id];
      delete state.questionAttachmentNotes[id];
      delete state.questionFiles[id];
      delete state.questionFileObjects[id];
      delete state.selectedChoices[id];
      toast(status === 'bypassed' ? 'Question bypassed.' : 'Answer saved.');
      render();
    }

    async function duplicateSection(id) {
      await api(`/api/sections/${id}/duplicate`, {method:'POST', body:'{}'});
      toast('Section duplicated.');
      render();
    }

    async function markSectionComplete(id) {
      const section = state.sections.find(item => item.id === id);
      const draft = cloneSection(section || {});
      ensureDraftStepIds(draft);
      const issues = sectionQaIssues(draft);
      if (issues.length) {
        state.currentSection = id;
        state.sectionDraftId = id;
        state.sectionDraft = draft;
        state.sectionStepIndexes[id] = issues[0].stepIndex;
        state.sectionQaFocus = issues[0].issueId;
        toast(`${issues.length} QA items must be resolved or bypassed.`);
        go('section', id);
        return;
      }
      try {
        await api(`/api/sections/${id}/complete`, {method:'POST', body:'{}'});
        toast('Section complete.');
        render();
      } catch (err) {
        toast('Section has unresolved QA items.');
        go('section', id);
      }
    }

    async function deleteSection(id) {
      const section = state.sections.find(item => item.id === id);
      const label = section?.title || 'this section';
      if (!confirm(`Delete ${label}? Packages already built from snapshots will not be changed.`)) return;
      await api(`/api/sections/${id}/delete`, {method:'POST', body:'{}'});
      delete state.sectionMediaOptions[id];
      if (state.currentSection === id) {
        state.currentSection = null;
        state.sectionDraft = null;
        state.sectionDraftId = null;
        await loadData();
        go('sections');
      } else {
        await loadData();
        toast('Section deleted.');
        render(false);
      }
    }

    async function saveSection(statusOverride = null) {
      const draft = collectSectionDraft();
      if (!draft) return;
      ensureDraftStepIds(draft);
      if (statusOverride === 'Complete') {
        const issues = sectionQaIssues(draft);
        if (issues.length) {
          draft.status = 'Needs Edit';
          state.sectionStepIndexes[draft.id] = issues[0].stepIndex;
          state.sectionQaFocus = issues[0].issueId;
          await api(`/api/sections/${draft.id}/update`, {
            method: 'POST',
            body: JSON.stringify({section: draft})
          });
          state.sectionDraft = cloneSection(draft);
          toast(`${issues.length} QA items must be resolved or bypassed.`);
          await loadData();
          render(false);
          return;
        }
      }
      if (statusOverride) draft.status = statusOverride;
      const result = await api(`/api/sections/${draft.id}/update`, {
        method: 'POST',
        body: JSON.stringify({section: draft})
      });
      if (result.blocked_qa && result.blocked_qa.length) {
        draft.status = 'Needs Edit';
        state.sectionStepIndexes[draft.id] = result.blocked_qa[0].step_index || 0;
        state.sectionQaFocus = result.blocked_qa[0].issue_id || null;
        toast(`${result.blocked_qa.length} QA items must be resolved or bypassed.`);
      } else {
        toast(statusOverride === 'Complete' ? 'Section completed.' : 'Section saved.');
      }
      state.sectionDraft = cloneSection(result.section || draft);
      await loadData();
      render(false);
    }

    function selectEditorStep(idx) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const next = Math.min(Math.max(Number(idx) || 0, 0), draft.steps.length - 1);
      state.sectionStepIndexes[draft.id] = next;
      state.sectionMediaPicker.open = false;
      state.sectionMediaPicker.replaceIndex = null;
      state.sectionMediaPicker.mode = 'append';
      renderSectionEditor();
      autosizeTextareas();
    }

    function addEditorStep() {
      const draft = collectSectionDraft();
      if (!draft) return;
      draft.steps.push(blankStep());
      state.sectionStepIndexes[draft.id] = draft.steps.length - 1;
      renderSectionEditor();
      autosizeTextareas();
    }

    function removeEditorStep(idx) {
      const draft = collectSectionDraft();
      if (!draft || draft.steps.length <= 1) {
        toast('A section needs at least one step.');
        return;
      }
      draft.steps.splice(idx, 1);
      state.sectionStepIndexes[draft.id] = Math.min(idx, draft.steps.length - 1);
      renderSectionEditor();
      autosizeTextareas();
    }

    function duplicateEditorStep(idx) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const clone = cloneSection(draft.steps[idx]);
      clone.step_id = newStepId();
      clone.qa_resolutions = {};
      draft.steps.splice(idx + 1, 0, clone);
      state.sectionStepIndexes[draft.id] = idx + 1;
      renderSectionEditor();
      autosizeTextareas();
    }

    function moveEditorStep(idx, dir) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const next = idx + dir;
      if (next < 0 || next >= draft.steps.length) return;
      const [item] = draft.steps.splice(idx, 1);
      draft.steps.splice(next, 0, item);
      state.sectionStepIndexes[draft.id] = next;
      renderSectionEditor();
      autosizeTextareas();
    }

    function nonEmptyValue(value) {
      const text = String(value || '').trim();
      if (!text || ['none', 'not required', 'n/a', 'na'].includes(text.toLowerCase())) return '';
      return text;
    }

    function combineTorque(first, second) {
      return nonEmptyValue(first.torque) || nonEmptyValue(second.torque) || first.torque || second.torque || '';
    }

    function mergeStepTiming(combined, first, second) {
      const numberValues = values => values.filter(value => value !== null && value !== undefined && value !== '').map(Number).filter(Number.isFinite);
      const starts = numberValues([first.start_time, second.start_time]);
      const ends = numberValues([first.end_time, second.end_time]);
      const targets = numberValues([first.target_time, second.target_time]);
      if (starts.length) combined.start_time = Math.min(...starts);
      if (ends.length) combined.end_time = Math.max(...ends);
      if (targets.length) combined.target_time = Math.max(...targets);
      combined.timing_confidence = first.timing_confidence || second.timing_confidence;
      combined.timing_source = first.timing_source || second.timing_source;
    }

    function combineSteps(first, second) {
      ensureStepIdentity(first);
      ensureStepIdentity(second);
      const combined = cloneSection(first);
      combined.text = combineTextValues([first.text, second.text], ' ');
      combined.tools = combineTextValues([first.tools, second.tools], ', ');
      combined.torque = combineTorque(first, second);
      combined.quality = combineTextValues([first.quality, second.quality], ' ');
      combined.safety = combineTextValues([first.safety, second.safety], ' ');
      combined.parts = uniqueList([...(first.parts || []), ...(second.parts || [])]);
      combined.media = uniqueList([...(first.media || []), ...(second.media || [])]);
      combined.media_settings = normalizeStepMediaSettings({
        media: combined.media,
        media_settings: {
          ...(first.media_settings || {}),
          ...(second.media_settings || {})
        }
      });
      combined.media_candidates = [...(first.media_candidates || []), ...(second.media_candidates || [])].slice(0, 12);
      combined.qa_resolutions = {...(first.qa_resolutions || {}), ...(second.qa_resolutions || {})};
      delete combined.qa_resolutions.merge_candidate;
      combined.merge_candidate = false;
      combined.merge_recommendation = null;
      const confidenceRank = {manual: 5, high: 4, medium: 3, low: 2, fallback: 1, missing: 0};
      combined.media_match_confidence = [first.media_match_confidence, second.media_match_confidence].sort((a, b) => (confidenceRank[b] || 0) - (confidenceRank[a] || 0))[0] || '';
      combined.media_match_score = Math.max(Number(first.media_match_score || 0), Number(second.media_match_score || 0));
      combined.media_fallback = combined.media_match_confidence === 'fallback' || (first.media_fallback && second.media_fallback && combined.media.length <= 1);
      if (!combined.media_fallback) combined.media_fallback_reason = '';
      mergeStepTiming(combined, first, second);
      return combined;
    }

    function mergeEditorStep(idx, dir) {
      const draft = collectSectionDraft();
      if (!draft) return;
      const targetIndex = dir < 0 ? idx - 1 : idx;
      const sourceIndex = dir < 0 ? idx : idx + 1;
      if (targetIndex < 0 || sourceIndex >= draft.steps.length) return;
      const combined = combineSteps(draft.steps[targetIndex], draft.steps[sourceIndex]);
      draft.steps.splice(targetIndex, 2, combined);
      state.sectionStepIndexes[draft.id] = targetIndex;
      state.sectionQaFocus = null;
      renderSectionEditor();
      autosizeTextareas();
      toast('Steps merged.');
    }

    function splitPointForText(text, cursor) {
      const cleaned = String(text || '');
      if (cursor && cursor > 8 && cursor < cleaned.length - 8) return cursor;
      const midpoint = Math.floor(cleaned.length / 2);
      const matches = [...cleaned.matchAll(/[.!?]\s+/g)].map(match => match.index + match[0].length);
      if (matches.length) return matches.sort((a, b) => Math.abs(a - midpoint) - Math.abs(b - midpoint))[0];
      const before = cleaned.lastIndexOf(' ', midpoint);
      const after = cleaned.indexOf(' ', midpoint);
      if (before > 8) return before;
      if (after > 8 && after < cleaned.length - 8) return after;
      return midpoint;
    }

    function splitEditorStep(idx) {
      const textarea = document.getElementById('step-text');
      const cursor = textarea ? textarea.selectionStart : null;
      const draft = collectSectionDraft();
      if (!draft) return;
      const step = draft.steps[idx];
      ensureStepIdentity(step);
      const text = String(step.text || '');
      if (text.trim().length < 18) {
        toast('Step text is too short to split cleanly.');
        return;
      }
      const splitAt = splitPointForText(text, cursor);
      const firstText = text.slice(0, splitAt).trim();
      const secondText = text.slice(splitAt).trim();
      if (!firstText || !secondText) {
        toast('Could not find a clean split point.');
        return;
      }
      const media = normalizeMediaList(step.media);
      const nextStep = {
        ...blankStep(),
        text: secondText,
        tools: step.tools || '',
        torque: step.torque || '',
        media: media.slice(1),
        media_settings: normalizeStepMediaSettings({media: media.slice(1), media_settings: step.media_settings || {}}),
        timing_source: step.timing_source,
        timing_confidence: step.timing_confidence,
        source_step_id: step.source_step_id,
        raw_transcript: step.raw_transcript,
        boundary_reason: 'manual split',
        transcript_confidence: step.transcript_confidence
      };
      step.text = firstText;
      step.media = media.slice(0, 1);
      step.media_settings = normalizeStepMediaSettings(step);
      step.qa_resolutions = {};
      step.merge_candidate = false;
      step.merge_recommendation = null;
      if (Number.isFinite(Number(step.start_time)) && Number.isFinite(Number(step.end_time)) && Number(step.end_time) > Number(step.start_time)) {
        const midpoint = Number(step.start_time) + (Number(step.end_time) - Number(step.start_time)) / 2;
        nextStep.start_time = Number(midpoint.toFixed(2));
        nextStep.end_time = step.end_time;
        nextStep.target_time = nextStep.end_time;
        step.end_time = Number(midpoint.toFixed(2));
        step.target_time = step.end_time;
      }
      draft.steps.splice(idx + 1, 0, nextStep);
      state.sectionStepIndexes[draft.id] = idx + 1;
      state.sectionQaFocus = null;
      renderSectionEditor();
      autosizeTextareas();
      toast('Step split.');
    }

    async function createPackageDraft(sectionIds = []) {
      const firstSection = state.sections.find(section => section.id === sectionIds[0]) || {};
      const payload = {
        title: firstSection.product ? `${firstSection.product} Instructions` : 'Instruction Package',
        product: firstSection.product || '',
        work_area: firstSection.work_area || '',
        station: firstSection.station || '',
        status: 'Draft',
        section_ids: sectionIds
      };
      const result = await api('/api/packages', {method:'POST', body: JSON.stringify(payload)});
      state.currentPackageData = result.package;
      state.packageDraftId = result.package.id;
      state.packageDraft = [...(result.package.section_ids || [])];
      state.packageDraftMeta = {
        title: result.package.title || '',
        product: result.package.product || '',
        work_area: result.package.work_area || '',
        station: result.package.station || '',
        status: result.package.status || 'Draft'
      };
      toast('Package draft created.');
      await loadData();
      go('package_edit', result.package.id);
    }

    function newPackageDraft() {
      createPackageDraft();
    }

    async function loadPackageDraft(id) {
      const result = await api(`/api/packages/${id}`);
      const packageData = result.package;
      state.currentPackageData = packageData;
      state.packageDraftId = packageData.id;
      state.packageDraft = [...(packageData.section_ids || [])];
      state.packageDraftMeta = {
        title: packageData.title || '',
        product: packageData.product || '',
        work_area: packageData.work_area || '',
        station: packageData.station || '',
        status: packageData.status || 'Draft'
      };
      go('package_edit', packageData.id);
    }

    function togglePackageSectionPreview(id) {
      collectPackageDraftMeta();
      state.expandedPackageSections[id] = !state.expandedPackageSections[id];
      renderPackageEditor();
    }

    async function addToPackage(id) {
      const section = state.sections.find(s => s.id === id);
      if (!section || section.status !== 'Complete') {
        toast('Only complete sections can be added to packages.');
        return;
      }
      if (state.currentPage !== 'package_edit' || !state.packageDraftId) {
        await createPackageDraft([id]);
        return;
      }
      collectPackageDraftMeta();
      if (state.packageDraft.includes(id)) {
        toast('Section is already in this package.');
        renderPackageEditor();
        return;
      }
      state.packageDraft.push(id);
      if (!state.packageDraftMeta.product) state.packageDraftMeta.product = section.product || '';
      if (!state.packageDraftMeta.work_area) state.packageDraftMeta.work_area = section.work_area || '';
      if (!state.packageDraftMeta.station) state.packageDraftMeta.station = section.station || '';
      if (!state.packageDraftMeta.title) state.packageDraftMeta.title = section.product ? `${section.product} Instructions` : 'Instruction Package';
      renderPackageEditor();
    }

    function removePackage(idx) {
      collectPackageDraftMeta();
      state.packageDraft.splice(idx, 1);
      renderPackageEditor();
    }

    function movePackage(idx, dir) {
      collectPackageDraftMeta();
      const next = idx + dir;
      if (next < 0 || next >= state.packageDraft.length) return;
      const [item] = state.packageDraft.splice(idx, 1);
      state.packageDraft.splice(next, 0, item);
      renderPackageEditor();
    }

    async function savePackage(next = 'edit') {
      const meta = collectPackageDraftMeta();
      const payload = {
        ...meta,
        title: meta.title || 'Instruction Package',
        section_ids: state.packageDraft
      };
      const result = state.packageDraftId
        ? await api(`/api/packages/${state.packageDraftId}/update`, {method:'POST', body: JSON.stringify(payload)})
        : await api('/api/packages', {method:'POST', body: JSON.stringify(payload)});
      state.packageDraftId = result.package.id;
      toast('Package saved.');
      await loadData();
      if (next === 'preview') {
        go('package_view', result.package.id);
      } else if (next === 'print') {
        go('package_print', result.package.id);
      } else {
        go('package_edit', result.package.id);
      }
    }

    async function snapshotPackage(id) {
      await api(`/api/packages/${id}/snapshot`, {method:'POST', body:'{}'});
      toast('Package snapshot refreshed from current sections.');
      await loadData();
      render(false);
    }

    async function deletePackage(id) {
      if (!confirm('Delete this package? Sections and media will be retained.')) return;
      await api(`/api/packages/${id}/delete`, {method:'POST', body:'{}'});
      toast('Package deleted.');
      state.packageDraftId = state.packageDraftId === id ? null : state.packageDraftId;
      if (state.currentPackage === id || state.currentPage === 'package_view' || state.currentPage === 'package_edit' || state.currentPage === 'package_print') {
        go('package');
      } else {
        await loadData();
        render(false);
      }
    }

    document.addEventListener('scroll', () => {
      state.lastScrollAt = Date.now();
    }, true);

    window.addEventListener('hashchange', () => {
      invalidateRender();
      render();
    });
    seed().catch(err => {
      const root = document.getElementById('root');
      if (root) {
        root.innerHTML = `<div class="empty">App failed to load: ${esc(err.message || err)}</div>`;
      }
      console.error(err);
    });

    async function seed() {
      readHash();
      await loadData();
      startPolling();
      render(false);
    }
