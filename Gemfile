source "https://rubygems.org"

# The `github-pages` gem pins an older Jekyll 3.x dependency stack that is
# incompatible with modern Ruby/Jekyll installs (e.g. Ruby 4.0 + Jekyll 4.4).
# It is only needed to match GitHub's *build* exactly; for local preview we
# just use the modern stack already installed. Deploying to GitHub Pages is
# unaffected, because GitHub builds using its own supported gems.
gem "jekyll", "~> 4.4"
gem "minima", "~> 2.5"   # theme used in _config.yml (same minor as Pages)
gem "jekyll-feed"        # plugin used in _config.yml

# Required to run `jekyll serve` on Ruby 3.0+
gem "webrick"

# Windows and JRuby do not include zoneinfo files, so bundle the tzinfo-data gem
# and associated library.
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end


