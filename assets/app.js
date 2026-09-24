/**
 * LoanGuard AI - Micro-interactions & UI enhancements
 * Vanilla JavaScript only.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Smooth scrolling for anchor links that point to IDs
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // Animate progress bars on reveal
    const progressBars = document.querySelectorAll(".lg-meter-fill");
    progressBars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = "0%";
        setTimeout(() => {
            bar.style.width = width;
        }, 150);
    });

    // Brand watermark
    console.log(
        "%c🛡️ LoanGuard AI %c Intelligent Credit Risk & Loan Default Prediction",
        "background:#FF6B00; color:#fff; font-size:12px; padding:4px 8px; border-radius:4px; font-weight:bold;",
        "color:#172033; font-size:12px; font-weight:600;"
    );
});
