/** UI-only translation: raw API output remains available in the collapsible JSON. */
export function presentModelWarning(warning: string): string {
  const normalized = warning.toLowerCase();
  if (normalized.includes('document_level_only')) {
    return 'M1 chỉ phát hiện dấu hiệu lỗi ở cấp độ toàn bài; không xác định hoặc sửa từng từ bị lỗi.';
  }
  if (normalized.includes('description') && /empty|missing|blank|not provided|unavailable/.test(normalized)) {
    return 'Thiếu phần mô tả ngắn. Kết quả M3 có thể thiếu thông tin từ nhánh tiêu đề–mô tả.';
  }
  if (normalized.includes('pyvi') || normalized.includes('experimental')) {
    return 'M1 dùng phân đoạn PyVi thử nghiệm; kết quả có thể khác với định dạng phân đoạn gốc.';
  }
  // Unknown technical warnings are kept visible, never silently discarded.
  return `Ghi chú kỹ thuật từ mô hình: ${warning}`;
}
