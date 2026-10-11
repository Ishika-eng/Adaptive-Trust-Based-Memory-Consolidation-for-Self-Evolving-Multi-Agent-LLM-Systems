export default function Logo({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" role="img" aria-label="ATMC logo">
      <defs>
        <linearGradient id="atmc-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#6366f1" /><stop offset="1" stopColor="#4338ca" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8" fill="url(#atmc-g)" />
      <g stroke="#fff" strokeWidth="1.8" strokeLinecap="round" fill="none"><path d="M10 21l6-10 6 10" /><path d="M12.5 17h7" /></g>
      <g fill="#fff"><circle cx="10" cy="21" r="2.2" /><circle cx="16" cy="11" r="2.2" /><circle cx="22" cy="21" r="2.2" /></g>
    </svg>
  )
}
