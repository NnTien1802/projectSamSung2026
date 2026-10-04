import type { ArticleForm } from './api';

/** Fictional teaching examples, not ground-truth labels or a model benchmark. */
export interface DemoCase {
  id: string;
  title: string;
  description: string;
  form: ArticleForm;
}

export const DEMO_CASES: DemoCase[] = [
  {
    id: 'neutral',
    title: 'Bài báo trung tính',
    description: 'Minh họa một bài báo có tiêu đề và nội dung cùng chủ đề.',
    form: {
      title: 'Người dân tham gia hoạt động trồng cây tại địa phương',
      description: 'Hoạt động trồng cây được tổ chức với sự tham gia của người dân và tình nguyện viên.',
      articleText: 'Sáng nay, nhiều người dân và tình nguyện viên tham gia trồng cây tại một khu vực công cộng. Ban tổ chức cho biết hoạt động nhằm nâng cao ý thức bảo vệ môi trường. Các nhóm tham gia đã chuẩn bị cây giống, dụng cụ và phân chia khu vực trước khi bắt đầu.',
      articleSegmented: '', m1Mode: 'skip',
    },
  },
  {
    id: 'typos',
    title: 'M1 · Văn bản chứa lỗi cố ý',
    description: 'Chứa một số lỗi gõ và lỗi chính tả; tự chọn chế độ PyVi thử nghiệm.',
    form: {
      title: 'Người dân tham gia buổi trồng cây tại công viên',
      description: 'Các tình nguyện viên cùng trồng cây và thu gom rác sau hoạt động.',
      articleText: 'Sáng nay, nhìu ngườii dân tham gia hoạt động trồng cây tại công viên. Ban tổ chức cho biết chương trình hướng tới việc nâng cao ý thức bảo vệ môi trương. Các tình nguyện viên đã chuẩn bị cây giống và dung cụ trước khi bắt đầu.',
      articleSegmented: '', m1Mode: 'pyvi_experimental',
    },
  },
  {
    id: 'headline',
    title: 'M2 · Tiêu đề mang phong cách giật tít',
    description: 'Ví dụ minh họa cách viết tiêu đề gây tò mò, không phải nhãn clickbait đã xác minh.',
    form: {
      title: 'Bạn sẽ không tin điều xảy ra ở công viên sáng nay!',
      description: 'Nhiều tình nguyện viên tham gia hoạt động trồng cây tại công viên.',
      articleText: 'Sáng nay, nhóm tình nguyện tại địa phương tổ chức hoạt động trồng cây tại công viên. Người tham gia nhận cây giống từ ban tổ chức rồi trồng vào những khu vực đã được phân chia. Hoạt động kết thúc trước buổi trưa.',
      articleSegmented: '', m1Mode: 'skip',
    },
  },
  {
    id: 'mismatch',
    title: 'M3 · Tiêu đề và nội dung khác chủ đề',
    description: 'Tiêu đề về trường học nhưng phần mô tả và nội dung nói về hoạt động trồng cây.',
    form: {
      title: 'Trường học tại địa phương công bố lịch học mới',
      description: 'Các tình nguyện viên đã tham gia hoạt động trồng cây tại công viên vào sáng nay.',
      articleText: 'Sáng nay, nhóm tình nguyện viên đã cùng người dân trồng cây tại một công viên trong thành phố. Ban tổ chức chuẩn bị cây giống và dụng cụ, đồng thời phân chia khu vực trồng cây cho từng nhóm. Hoạt động nhằm nâng cao ý thức bảo vệ môi trường.',
      articleSegmented: '', m1Mode: 'skip',
    },
  },
];
