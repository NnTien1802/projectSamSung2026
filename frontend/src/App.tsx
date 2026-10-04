import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { analyze, getHealth, type AnalyzeResponse, type ArticleForm, type HealthResponse, type M1Mode, type ModuleResponse } from './api';
import { DEMO_CASES } from './demo_cases';
import { presentModelWarning } from './presentation';

type AnalysisState = 'idle' | 'loading' | 'complete' | 'error';
const INIT_FORM: ArticleForm = {
  title: '', description: '', articleText: '', articleSegmented: '', m1Mode: 'skip',
};

const labels: Record<string, string> = {
  NO_ERROR: 'Không phát hiện lỗi', HAS_ERROR: 'Có dấu hiệu lỗi',
  NORMAL: 'Không clickbait', CLICKBAIT: 'Có dấu hiệu clickbait',
  CONSISTENT: 'Nội dung nhất quán', INCONSISTENT: 'Có dấu hiệu không nhất quán',
};

function percent(n: number): string { return `${(n * 100).toFixed(1).replace('.', ',')}%`; }
function numberOrNull(v: unknown): number | null { return typeof v === 'number' && Number.isFinite(v) ? v : null; }

function ResultCard({ entry, module, title, subtitle }: {
  entry: ModuleResponse; module: 1 | 2 | 3; title: string; subtitle: string;
}) {
  const result = entry.result;
  const probKey = module === 1 ? 'probability_error' : module === 2 ? 'probability_clickbait' : 'probability_consistent';
  const probability = numberOrNull(result?.[probKey]);
  const threshold = numberOrNull(result?.threshold);
  const meanings = module === 1 ? 'có lỗi' : module === 2 ? 'clickbait' : 'nhất quán';
  const warnings = Array.isArray(result?.warnings) ? result.warnings.filter((v): v is string => typeof v === 'string') : [];

  return <article className="result-card">
    <div className="result-top">
      <div className="result-icon">M{module}</div>
      <div><h3>{title}</h3><p>{subtitle}</p></div>
      <span className={`result-status ${entry.status === 'completed' ? 'success' : 'neutral'}`}>
        {entry.status === 'completed' ? 'Đã phân tích' : entry.status === 'skipped' ? 'Bỏ qua' : entry.status}
      </span>
    </div>
    {entry.status === 'skipped' ? <div className="skip-box">
      M1 chưa chạy. Bạn có thể cung cấp văn bản đã phân đoạn hoặc chủ động chọn chế độ PyVi thử nghiệm.
    </div> : result ? <>
      <div className="prediction-title">{labels[result.label] ?? result.label}</div>
      <div className="probability-row">
        <span>Xác suất mô hình: {meanings}</span>
        <strong>{probability !== null ? percent(probability) : 'Không có dữ liệu'}</strong>
      </div>
      {probability !== null && <div className="track" role="img" aria-label={`Xác suất ${meanings}: ${percent(probability)}`}>
        <div className="track-fill" style={{ width: `${Math.max(0, Math.min(100, probability * 100))}%` }} />
        {threshold !== null && <i className="threshold-line" style={{ left: `${threshold * 100}%` }} title={`Ngưỡng ${percent(threshold)}`} />}
      </div>}
      <div className="result-metadata">
        {threshold !== null && <span>Ngưỡng cố định: {threshold.toFixed(2)}</span>}
        {module === 1 && numberOrNull(result.chunk_count) !== null && <span>Số đoạn: {String(result.chunk_count)}</span>}
        {entry.input_mode && <span>Đầu vào: {entry.input_mode === 'presegmented' ? 'Phân đoạn có sẵn' : 'PyVi thử nghiệm'}</span>}
      </div>
      {entry.warning && <div className="inline-warning" role="note">M1 sử dụng PyVi thử nghiệm. Kết quả có thể khác với phân đoạn gốc trong nghiên cứu.</div>}
      {warnings.map((w, i) => <div className="inline-warning" role="note" key={i}>{presentModelWarning(w)}</div>)}
      {module === 3 && !result.branches && <p className="muted small">Không có chi tiết từng nhánh trong phản hồi API.</p>}
      <details className="raw-details"><summary>Xem dữ liệu kết quả</summary><pre>{JSON.stringify(result, null, 2)}</pre></details>
    </> : <p className="muted">API chưa trả về kết quả cho module này.</p>}
  </article>;
}

export default function App() {
  const [form, setForm] = useState<ArticleForm>(INIT_FORM);
  const [demoId, setDemoId] = useState(DEMO_CASES[0].id);
  const [activeDemoTitle, setActiveDemoTitle] = useState('');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [state, setState] = useState<AnalysisState>('idle');
  const [error, setError] = useState('');
  const [results, setResults] = useState<AnalyzeResponse | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const resultRef = useRef<HTMLElement | null>(null);

  const refreshHealth = useCallback(async () => {
    try { setHealth(await getHealth()); setHealthError(false); }
    catch { setHealth(null); setHealthError(true); }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    getHealth(controller.signal).then((v) => { setHealth(v); setHealthError(false); })
      .catch((err: unknown) => { if (!(err instanceof DOMException && err.name === 'AbortError')) setHealthError(true); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (results && state === 'complete') resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [results, state]);

  useEffect(() => () => requestRef.current?.abort(), []);

  const update = <K extends keyof ArticleForm>(field: K, value: ArticleForm[K]) => {
    setForm((old) => ({ ...old, [field]: value }));
  };

  const valid = form.title.trim().length > 0 && form.title.length <= 1000 &&
    form.articleText.trim().length > 0 && form.articleText.length <= 120000 &&
    form.description.length <= 5000 &&
    (form.m1Mode !== 'presegmented' || (form.articleSegmented.trim().length > 0 && form.articleSegmented.length <= 120000));

  const incompleteReason = !form.title.trim() ? 'Nhập tiêu đề bài báo.' :
    !form.articleText.trim() ? 'Nhập nội dung bài báo.' :
    form.m1Mode === 'presegmented' && !form.articleSegmented.trim() ? 'Nhập văn bản đã phân đoạn cho M1 hoặc chọn chế độ khác.' : '';

  const surfaceWarning = form.m1Mode === 'presegmented' && !!form.articleSegmented.trim() && !!form.articleText.trim() &&
    form.articleText.normalize('NFC').replace(/[\s_]+/gu, '') !==
    form.articleSegmented.normalize('NFC').replace(/[\s_]+/gu, '');

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!valid || state === 'loading') return;
    setError(''); setResults(null); setState('loading');
    const controller = new AbortController(); requestRef.current = controller;
    try {
      const output = await analyze(form, controller.signal);
      setResults(output); setState('complete');
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      setError(err instanceof Error ? err.message : 'Đã xảy ra lỗi không xác định.');
      setState('error');
    } finally { requestRef.current = null; }
  }

  function reset() {
    if (state === 'loading') return;
    setForm(INIT_FORM); setActiveDemoTitle(''); setResults(null); setError(''); setState('idle');
  }

  function applyDemo() {
    if (state === 'loading') return;
    const sample = DEMO_CASES.find((demo) => demo.id === demoId) ?? DEMO_CASES[0];
    setForm({ ...sample.form }); setActiveDemoTitle(sample.title);
    setResults(null); setError(''); setState('idle');
  }

  return <div className="page-shell">
    <header className="site-header">
      <div className="container header-inner">
        <div className="brand"><span className="brand-mark">S</span><div><strong>projectSamSung</strong><span>Phân tích báo chí tiếng Việt · V1</span></div></div>
        <div className="header-right"><span className="desktop-only">Hệ thống nghiên cứu · Local</span><button type="button" className="health-button" onClick={refreshHealth} title="Kiểm tra lại backend">
          <span className={`status-dot ${health?.models_ready ? 'online' : 'offline'}`} />
          {health?.models_ready ? 'API sẵn sàng' : healthError ? 'Chưa kết nối API' : 'Đang khởi tạo'}
          <span aria-hidden="true">↻</span>
        </button></div>
      </div>
    </header>

    <main className="container main">
      <section className="intro"><div className="eyebrow"><span className="eyebrow-dot"/> HỆ THỐNG PHÂN TÍCH BÁO CHÍ TIẾNG VIỆT</div>
        <h1>Một bài báo.<br/><em>Ba góc nhìn phân tích.</em></h1>
        <p>Nhập tiêu đề, mô tả và nội dung để kiểm tra dấu hiệu lỗi văn bản, clickbait và tính nhất quán giữa tiêu đề với bài viết bằng ba module AI độc lập.</p>
      </section>

      {(healthError || (health && !health.models_ready)) && <div className="backend-notice" role="status">
        <div><strong>{healthError ? 'Chưa kết nối được với FastAPI' : 'Mô hình đang khởi tạo'}</strong>
          <p>{healthError ? 'Hãy chạy backend trên cổng 8000, sau đó kiểm tra lại kết nối.' : 'Đợi backend nạp các mô hình xong rồi thử lại.'}</p>
        </div>
        <button type="button" className="secondary-btn" onClick={refreshHealth}>Kiểm tra lại</button>
      </div>}

      <div className="workspace">
        <section className="form-panel" aria-labelledby="form-title">
          <div className="panel-head"><div><span className="step-label">01 · DỮ LIỆU ĐẦU VÀO</span><h2 id="form-title">Nội dung bài báo</h2></div></div>
          <div className="demo-panel"><label htmlFor="demo-select">Kịch bản minh họa</label>
            <div className="demo-controls"><select id="demo-select" value={demoId} disabled={state === 'loading'} onChange={(e) => setDemoId(e.target.value)}>
              {DEMO_CASES.map((demo) => <option key={demo.id} value={demo.id}>{demo.title}</option>)}
            </select><button type="button" className="secondary-btn" onClick={applyDemo} disabled={state === 'loading'}>Điền bài mẫu</button></div>
            <p>{DEMO_CASES.find((demo) => demo.id === demoId)?.description} Các mẫu được tự tạo để trình diễn; mô hình không được bảo đảm trả về một nhãn cụ thể.</p>
            {activeDemoTitle && <span className="demo-active">Đã điền: {activeDemoTitle}. Bạn có thể sửa nội dung trước khi phân tích.</span>}
          </div>
          <form onSubmit={submit}>
            <div className="field"><label htmlFor="title">Tiêu đề <span className="required">*</span></label>
              <input id="title" type="text" placeholder="Nhập tiêu đề bài báo..." maxLength={1000} required value={form.title} onChange={e => update('title', e.target.value)} />
              <span className="counter">{form.title.length.toLocaleString('vi-VN')} / 1.000</span></div>
            <div className="field"><label htmlFor="description">Mô tả ngắn <span className="optional">(khuyến nghị cho M3)</span></label>
              <textarea id="description" placeholder="Nhập phần sapo hoặc mô tả ngắn..." rows={3} maxLength={5000} value={form.description} onChange={e => update('description', e.target.value)}/>
              <span className="counter">{form.description.length.toLocaleString('vi-VN')} / 5.000</span></div>
            <div className="field"><label htmlFor="article">Nội dung bài báo <span className="required">*</span></label>
              <textarea id="article" className="article-input" placeholder="Dán toàn bộ nội dung bài báo bằng tiếng Việt vào đây..." rows={9} maxLength={120000} required value={form.articleText} onChange={e => update('articleText', e.target.value)}/>
              <span className="counter">{form.articleText.length.toLocaleString('vi-VN')} / 120.000</span></div>

            <fieldset className="mode-fieldset"><legend>Đầu vào Module 1 <span className="optional">· Chọn cách xử lý</span></legend>
              <label className={`mode-option ${form.m1Mode === 'skip' ? 'selected' : ''}`}>
                <input type="radio" name="m1-mode" checked={form.m1Mode === 'skip'} onChange={() => update('m1Mode', 'skip' as M1Mode)} />
                <span className="radio-marker"/><span><strong>Chỉ chạy M2 và M3</strong><small>M1 bỏ qua nếu bạn chưa có văn bản đã phân đoạn.</small></span>
                <span className="mode-tag">Mặc định</span>
              </label>
              <label className={`mode-option ${form.m1Mode === 'presegmented' ? 'selected' : ''}`}>
                <input type="radio" name="m1-mode" checked={form.m1Mode === 'presegmented'} onChange={() => update('m1Mode', 'presegmented' as M1Mode)} />
                <span className="radio-marker"/><span><strong>Phân đoạn có sẵn</strong><small>Sử dụng đầu vào `article_segmented` tương thích với nghiên cứu.</small></span>
                <span className="mode-tag">Tiêu chuẩn</span>
              </label>
              <label className={`mode-option ${form.m1Mode === 'pyvi_experimental' ? 'selected' : ''}`}>
                <input type="radio" name="m1-mode" checked={form.m1Mode === 'pyvi_experimental'} onChange={() => update('m1Mode', 'pyvi_experimental' as M1Mode)} />
                <span className="radio-marker"/><span><strong>Phân đoạn tự động bằng PyVi</strong><small>Chạy M1 trên bài báo thường. Kết quả chỉ dùng để thử nghiệm.</small></span>
                <span className="mode-tag warning-tag">Thử nghiệm</span>
              </label>
            </fieldset>
            {form.m1Mode === 'presegmented' && <div className="field segmented-field">
              <label htmlFor="segmented">Nội dung đã phân đoạn <span className="required">*</span></label>
              <textarea id="segmented" rows={5} placeholder="Ví dụ: người_dân đang đọc báo ." maxLength={120000} value={form.articleSegmented} onChange={e => update('articleSegmented', e.target.value)} />
              <span className="counter">{form.articleSegmented.length.toLocaleString('vi-VN')} / 120.000</span>
              {surfaceWarning && <div role="note" className="inline-warning">Văn bản thường và bản đã phân đoạn có nội dung khác nhau sau khi bỏ khoảng trắng/dấu _. Hãy bảo đảm chúng là cùng một bài báo.</div>}
            </div>}
            {form.m1Mode === 'pyvi_experimental' && <div className="experiment-note" role="note"><strong>Lưu ý tiền xử lý thử nghiệm</strong><p>Trên 60 mẫu validation, PyVi làm thay đổi nhãn M1 ở 8 mẫu. Kết quả chế độ này không tương đương với phân đoạn gốc và không dùng làm đánh giá mô hình chính thức.</p></div>}
            {!form.description.trim() && <p className="help-note">Mô tả đang để trống. M3 vẫn chạy nhưng có thể đưa ra cảnh báo về đầu vào thiếu mô tả.</p>}
            {incompleteReason && <p className="form-hint" role="status">{incompleteReason}</p>}
            <div className="form-actions"><button className="primary-btn" type="submit" disabled={!valid || state === 'loading'}>{state === 'loading' ? <><span className="spinner"/>Đang phân tích...</> : <>Phân tích bài báo <span aria-hidden="true">→</span></>}</button><button className="secondary-btn" type="button" disabled={state === 'loading'} onClick={reset}>Xóa dữ liệu</button></div>
            {state === 'error' && <div className="error-banner" role="alert">{error}</div>}
          </form>
        </section>

        <aside className="side-panel"><div className="panel-head"><div><span className="step-label">HỆ THỐNG AI</span><h2>Ba module độc lập</h2></div></div>
          <div className="module-blurb"><span className="module-badge">01</span><div><h3>Kiểm tra lỗi văn bản</h3><p>PhoBERT MIL · Dự đoán dấu hiệu lỗi ở cấp độ bài báo, không xác định hay sửa từng từ.</p><span className="threshold-tag">Ngưỡng 0.57</span></div></div>
          <div className="module-blurb"><span className="module-badge">02</span><div><h3>Phát hiện clickbait</h3><p>mBERT · Phân tích tiêu đề, độc lập với nội dung bài báo.</p><span className="threshold-tag">Ngưỡng 0.61</span></div></div>
          <div className="module-blurb"><span className="module-badge">03</span><div><h3>Kiểm tra tính nhất quán</h3><p>mBERT hybrid · Kết hợp tiêu đề–mô tả và tiêu đề–nội dung.</p><span className="threshold-tag">Ngưỡng 0.45</span></div></div>
          <div className="privacy-note"><strong>Chạy trên máy cục bộ</strong><p>Dữ liệu được gửi đến FastAPI đang chạy trên máy của bạn. Giao diện không lưu bài báo vào bộ nhớ trình duyệt hay gửi đến dịch vụ bên ngoài.</p></div>
        </aside>
      </div>

      <section className="results-section" aria-labelledby="results-title" ref={resultRef}>
        <div className="panel-head results-heading"><div><span className="step-label">02 · KẾT QUẢ</span><h2 id="results-title">Phân tích độc lập</h2></div>{results && <span className="completed-badge">Hoàn tất</span>}</div>
        {state === 'loading' && <div className="waiting"><span className="spinner dark"/><div><strong>Đang thực hiện suy luận...</strong><p>Máy chủ xử lý tuần tự các module trên một GPU. Vui lòng chờ.</p></div></div>}
        {results ? <>
          <div className="result-grid">
            <ResultCard entry={results.module1_spelling} module={1} title="Lỗi văn bản" subtitle="PhoBERT MIL" />
            <ResultCard entry={results.module2_clickbait} module={2} title="Clickbait" subtitle="mBERT · Tiêu đề" />
            <ResultCard entry={results.module3_consistency} module={3} title="Tính nhất quán" subtitle="mBERT · Hybrid" />
          </div>
          <p className="results-disclaimer">Ba module đưa ra kết quả độc lập, không tạo điểm chất lượng tổng hợp. Xác suất là đầu ra mô hình, không phải sự xác minh đúng/sai của toàn bộ bài báo.</p>
        </> : state !== 'loading' && <div className="empty-results"><div className="empty-symbol">◎</div><strong>Chưa có kết quả phân tích</strong><p>Điền thông tin bài báo, chọn chế độ đầu vào M1 và nhấn “Phân tích bài báo”.</p></div>}
      </section>
    </main>
    <footer className="footer"><div className="container footer-inner"><span>projectSamSung · Giao diện nghiên cứu V1</span><span>FastAPI · React · PhoBERT · mBERT</span></div></footer>
  </div>;
}
